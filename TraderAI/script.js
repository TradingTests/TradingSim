// TraderAI - Main script
console.log("TraderAI script loaded.");

// --- CONFIG ---
const ALL_SYMBOLS = {
    USDT: ['BTCUSDT', 'ETHUSDT', 'TONUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'NOTUSDT', 'SHIBUSDT', 'BNBUSDT', 'TRXUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT'],
    IRT: ['BTCIRT', 'ETHIRT', 'USDTIRT', 'TONIRT', 'XRPIRT', 'DOGEIRT', 'SHIBIRT', 'BNBIRT', 'TRXIRT', 'ADAIRT', 'AVAXIRT', 'LINKIRT']
};
let activeSymbols = ALL_SYMBOLS.IRT; // Default to IRT for now
const LOOP_INTERVAL = 15000; // 15 seconds
const LEARNING_RATE = 0.01;
const EXPLORATION_RATE = 0.1; // 10% chance of exploring

// --- DOM ELEMENTS ---
const startStopBtn = document.getElementById('start-stop-btn');
const tradeAmountInput = document.getElementById('trade-amount');
const cashValueSpan = document.getElementById('cash-value');
const tokenListUl = document.getElementById('token-list');
const logListUl = document.getElementById('log-list');
const portfolioChartCanvas = document.getElementById('portfolio-chart').getContext('2d');


// --- STATE ---
const wallet = {
    cash: 10000,
    tokens: {},
};
let model;
let mainLoopInterval = null;
let portfolioChart;
const portfolioHistory = []; // To track portfolio value over time
let tradesMadeLastCycle = []; // To calculate rewards on the next cycle

// --- API & TRADING LOGIC ---

/**
 * Fetches market data for a given symbol.
 * @param {string} symbol - The market symbol (e.g., 'BTCIRT').
 * @returns {Promise<Object|null>} A market data object or null on failure.
 */
async function fetchMarketData(symbol) {
    try {
        // Fetch order book for the specific symbol
        const orderbookResponse = await fetch(`https://apiv2.nobitex.ir/v3/orderbook/${symbol}`);
        if (!orderbookResponse.ok) {
            throw new Error(`Failed to fetch orderbook for ${symbol}`);
        }
        const orderbookData = await orderbookResponse.json();
        if (orderbookData.status !== 'ok') {
            throw new Error(`Orderbook status not ok for ${symbol}`);
        }

        // Fetch trades for the specific symbol
        const tradesResponse = await fetch(`https://apiv2.nobitex.ir/v2/trades/${symbol}`);
        if (!tradesResponse.ok) {
            throw new Error(`Failed to fetch trades for ${symbol}`);
        }
        const tradesData = await tradesResponse.json();
        const recentTrades = tradesData.trades.slice(0, 20);

        if (!orderbookData.bids?.[0] || !orderbookData.asks?.[0] || recentTrades.length === 0) {
            console.warn(`Incomplete data for ${symbol}.`);
            return null;
        }

        const bestBid = parseFloat(orderbookData.bids[0][0]);
        const bestAsk = parseFloat(orderbookData.asks[0][0]);

        if (isNaN(bestBid) || isNaN(bestAsk)) {
            console.warn(`Invalid bid/ask prices for ${symbol}.`);
            return null;
        }

        const price = (bestBid + bestAsk) / 2;
        return { symbol, price, recentTrades, bestBid, bestAsk };

    } catch (error) {
        console.error(`Error fetching market data for ${symbol}:`, error);
        return null;
    }
}

/**
 * Buys a token.
 * @param {string} symbol - The symbol of the token to buy.
 * @param {number} amount - The amount of cash to spend.
 * @param {number} price - The current price of the token.
 * @param {tf.Tensor} stateTensor - The state tensor at the time of decision.
 */
function buy(symbol, amount, price, stateTensor) {
    if (wallet.cash >= amount) {
        wallet.cash -= amount;
        const tokenAmount = amount / price;
        wallet.tokens[symbol] = (wallet.tokens[symbol] || 0) + tokenAmount;

        const tradeInfo = {
            id: Date.now(),
            symbol,
            action: 'buy',
            price,
            amount,
            stateTensor,
            timestamp: Date.now()
        };
        tradesMadeLastCycle.push(tradeInfo);
        addLog(`BOUGHT: ${tokenAmount.toFixed(6)} ${symbol} for ${amount} cash.`, tradeInfo);
    } else {
        addLog(`INFO: Not enough cash to buy ${symbol}.`);
        stateTensor.dispose(); // Clean up unused tensor
    }
}

/**
 * Sells a token.
 * @param {string} symbol - The symbol of the token to sell.
 * @param {number} amount - The amount of cash to receive.
 * @param {number} price - The current price of the token.
 * @param {tf.Tensor} stateTensor - The state tensor at the time of decision.
 */
function sell(symbol, amount, price, stateTensor) {
    const tokenAmountToSell = amount / price;
    if (wallet.tokens[symbol] && wallet.tokens[symbol] >= tokenAmountToSell) {
        wallet.tokens[symbol] -= tokenAmountToSell;
        wallet.cash += amount;
        const tradeInfo = {
            id: Date.now(),
            symbol,
            action: 'sell',
            price,
            amount,
            stateTensor,
            timestamp: Date.now()
        };
        tradesMadeLastCycle.push(tradeInfo);
        addLog(`SOLD: ${tokenAmountToSell.toFixed(6)} ${symbol} for ${amount} cash.`, tradeInfo);
    } else {
        addLog(`INFO: Not enough ${symbol} to sell.`);
        stateTensor.dispose(); // Clean up unused tensor
    }
}

// --- AI MODEL ---

/**
 * Creates and compiles the reinforcement learning model.
 */
function createModel() {
    const newModel = tf.sequential();
    // Input layer: Takes a flattened array of market state
    // State: [price, cash, bestBid, bestAsk, spread, last20TradeTypes...] (25 total)
    newModel.add(tf.layers.dense({ inputShape: [25], units: 32, activation: 'relu' }));
    // Hidden layer
    newModel.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    // Output layer: 2 actions (buy, sell)
    newModel.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Linear for Q-values

    newModel.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'meanSquaredError'
    });

    model = newModel;
    console.log("AI Model created successfully.");
}

/**
 * Preprocesses market and wallet data into a tensor for the model.
 * @param {Object} marketData - The market data from fetchMarketData.
 * @returns {tf.Tensor|null} A tensor representing the current state, or null if data is invalid.
 */
function preprocessData({ price, bestBid, bestAsk, recentTrades }) {
    if (!price || !bestBid || !bestAsk || !recentTrades || recentTrades.length === 0) {
        return null;
    }

    const avgPrice = recentTrades.reduce((acc, trade) => acc + parseFloat(trade.price), 0) / recentTrades.length;
    if (isNaN(avgPrice) || avgPrice === 0) return null;

    const normalizedPrice = price / avgPrice;
    const normalizedBestBid = bestBid / avgPrice;
    const normalizedBestAsk = bestAsk / avgPrice;
    const spread = (bestAsk - bestBid) / avgPrice;

    const normalizedCash = wallet.cash / 10000;

    const tradeTypes = recentTrades.map(trade => (trade.type === 'buy' ? 1 : -1));
    while (tradeTypes.length < 20) {
        tradeTypes.push(0);
    }

    const state = [normalizedPrice, normalizedCash, normalizedBestBid, normalizedBestAsk, spread, ...tradeTypes];
    return tf.tensor2d(state, [1, 25]);
}

/**
 * Uses the model to make a trading decision using an epsilon-greedy strategy.
 * @param {tf.Tensor} stateTensor - The preprocessed state tensor.
 * @returns {Promise<string>} The decision: 'buy' or 'sell'.
 */
async function makeDecision(stateTensor, symbol) {
    if (!model || !stateTensor) {
        return 'hold';
    }

    let action;
    // Epsilon-greedy strategy for exploration
    if (Math.random() < EXPLORATION_RATE) {
        addLog("Exploring: Making a random decision.");
        action = Math.random() < 0.5 ? 'buy' : 'sell';
    } else {
        // Exploit: Use the model's prediction
        action = tf.tidy(() => {
            const prediction = model.predict(stateTensor);
            const actionIndex = prediction.argMax(1).dataSync()[0];
            return actionIndex === 0 ? 'buy' : 'sell';
        });
    }

    // Guardrail: If the model decides to sell a token it doesn't have, force a buy.
    // This helps the AI learn in the beginning when its wallet is empty.
    if (action === 'sell' && (!wallet.tokens[symbol] || wallet.tokens[symbol] <= 0)) {
        addLog(`INFO: Overriding "sell" for ${symbol} as no tokens are held. Forcing "buy".`);
        return 'buy';
    }

    return action;
}

/**
 * Trains the model with a given state, action, and reward.
 * @param {tf.Tensor} stateTensor - The state in which the action was taken.
 * @param {string} action - The action taken ('buy' or 'sell').
 * @param {number} reward - The calculated reward for the action.
 */
async function trainModel(stateTensor, action, reward) {
    const actionIndex = (action === 'buy') ? 0 : 1;

    // Get the current Q-values for the state
    const currentQValues = model.predict(stateTensor);
    const qValues = currentQValues.dataSync();

    // Update the Q-value for the action that was taken
    qValues[actionIndex] = reward;

    const targetQValues = tf.tensor2d(qValues, [1, 2]);

    // Train the model
    await model.fit(stateTensor, targetQValues);

    addLog(`Trained model for ${action} with reward ${reward.toFixed(4)}.`);

    // Clean up tensors
    stateTensor.dispose();
    currentQValues.dispose();
    targetQValues.dispose();
}


// --- UI FUNCTIONS ---

function addLog(message, tradeInfo = null) {
    const li = document.createElement('li');
    li.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;

    if (tradeInfo) {
        const rewardContainer = document.createElement('span');
        const posButton = document.createElement('button');
        posButton.textContent = '+1';
        posButton.onclick = () => {
            trainModel(tradeInfo.stateTensor, tradeInfo.action, 1);
            rewardContainer.remove(); // Remove buttons after clicking
        };

        const negButton = document.createElement('button');
        negButton.textContent = '-1';
        negButton.onclick = () => {
            trainModel(tradeInfo.stateTensor, tradeInfo.action, -1);
            rewardContainer.remove();
        };

        rewardContainer.appendChild(posButton);
        rewardContainer.appendChild(negButton);
        li.appendChild(rewardContainer);
    }
    logListUl.prepend(li);
}

function updateUI(marketData) {
    cashValueSpan.textContent = wallet.cash.toFixed(2);

    tokenListUl.innerHTML = '';
    let totalTokenValue = 0;
    for (const symbol in wallet.tokens) {
        if (wallet.tokens[symbol] > 0) {
            const li = document.createElement('li');
            li.textContent = `${symbol}: ${wallet.tokens[symbol].toFixed(6)}`;
            tokenListUl.appendChild(li);

            // Use the latest market data to calculate total value
            const currentMarket = marketData.find(m => m.symbol === symbol);
            if (currentMarket) {
                totalTokenValue += wallet.tokens[symbol] * currentMarket.price;
            }
        }
    }

    const portfolioValue = wallet.cash + totalTokenValue;
    updateChart(portfolioValue);
}

function updateChart(newValue) {
    portfolioChart.data.labels.push(new Date());
    portfolioChart.data.datasets.forEach((dataset) => {
        dataset.data.push(newValue);
    });
    portfolioChart.update();
}


// --- MAIN LOOP ---
async function rewardAndTrain(marketData) {
    if (!marketData || marketData.length === 0) {
        // Can't calculate rewards without fresh data
        // Dispose tensors to prevent memory leaks
        tradesMadeLastCycle.forEach(trade => trade.stateTensor.dispose());
        tradesMadeLastCycle = [];
        return;
    }

    for (const trade of tradesMadeLastCycle) {
        const currentData = marketData.find(d => d.symbol === trade.symbol);
        if (currentData) {
            const priceNow = currentData.price;
            const priceThen = trade.price;
            let reward = 0;

            if (trade.action === 'buy') {
                // Positive reward if price went up
                reward = (priceNow - priceThen) / priceThen;
            } else if (trade.action === 'sell') {
                // Positive reward if price went down (avoided loss)
                reward = (priceThen - priceNow) / priceThen;
            }

            // Train the model with the calculated reward
            await trainModel(trade.stateTensor, trade.action, reward);
        } else {
            // If we can't find market data, we can't reward. Dispose tensor.
            trade.stateTensor.dispose();
        }
    }

    // Clear the array for the next cycle
    tradesMadeLastCycle = [];
}

async function runMainLoop() {
    try {
        addLog("Main loop: Starting cycle.");
        const tradeAmount = parseFloat(tradeAmountInput.value);
        if (isNaN(tradeAmount) || tradeAmount <= 0) {
            addLog("ERROR: Invalid trade amount.");
            return;
        }

        addLog("Main loop: Fetching market data...");
        const marketData = [];
        for (const symbol of activeSymbols) {
            const data = await fetchMarketData(symbol);
            if (data) {
                marketData.push(data);
            }
            await new Promise(resolve => setTimeout(resolve, 250)); // Rate-limiting delay
        }

        updateUI(marketData);

        if (marketData.length === 0) {
            addLog("Main loop: Could not fetch any market data. Retrying next cycle.");
            return;
        }
        addLog(`Main loop: Fetched data for ${marketData.length}/${activeSymbols.length} markets.`);

        await rewardAndTrain(marketData);

        addLog("Main loop: AI is making decisions...");
        for (const data of marketData) {
            const stateTensor = preprocessData(data);
            if (stateTensor) {
                const decision = await makeDecision(stateTensor.clone(), data.symbol);
                addLog(`AI decision for ${data.symbol}: ${decision}`);

                if (decision === 'buy') {
                    buy(data.symbol, tradeAmount, data.price, stateTensor);
                } else if (decision === 'sell') {
                    sell(data.symbol, tradeAmount, data.price, stateTensor);
                } else {
                    stateTensor.dispose();
                }
            }
        }

        updateUI(marketData);
        addLog("Main loop: Cycle finished.");
    } catch (error) {
        console.error("Error in main loop:", error);
        addLog(`ERROR in main loop: ${error.message}`);
    }
}

function start() {
    if (mainLoopInterval) return; // Already running
    addLog("Starting TraderAI...");
    startStopBtn.textContent = 'Stop';
    mainLoopInterval = setInterval(runMainLoop, LOOP_INTERVAL);
    runMainLoop(); // Run once immediately
}

function stop() {
    if (!mainLoopInterval) return; // Already stopped
    addLog("Stopping TraderAI...");
    clearInterval(mainLoopInterval);
    mainLoopInterval = null;
    startStopBtn.textContent = 'Start';
}

// --- INITIALIZATION ---
function init() {
    createModel();

    // Initialize the chart
    portfolioChart = new Chart(portfolioChartCanvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'second'
                    },
                    ticks: {
                        source: 'auto'
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });


    // Attach event listeners
    startStopBtn.addEventListener('click', () => {
        if (mainLoopInterval) {
            stop();
        } else {
            start();
        }
    });

    const applySettingsBtn = document.getElementById('apply-settings-btn');
    applySettingsBtn.addEventListener('click', applySettings);

    applySettings(); // Apply default settings on init
    addLog("TraderAI initialized. Apply settings and click 'Start' to begin.");
}

function applySettings() {
    stop(); // Stop the bot if it's running

    const targetCurrency = document.getElementById('target-currency').value;
    const initialCash = parseFloat(document.getElementById('initial-cash').value);

    if (isNaN(initialCash) || initialCash <= 0) {
        alert("Please enter a valid initial cash amount.");
        return;
    }

    activeSymbols = ALL_SYMBOLS[targetCurrency];
    wallet.cash = initialCash;
    wallet.tokens = {};
    tradesMadeLastCycle = [];

    // Reset UI
    updateUI([]);
    portfolioHistory.length = 0;
    if (portfolioChart) {
        portfolioChart.data.labels = [];
        portfolioChart.data.datasets.forEach((dataset) => {
            dataset.data = [];
        });
        portfolioChart.update();
    }

    logListUl.innerHTML = '';
    addLog(`Settings applied. Target: ${targetCurrency}, Initial Cash: ${initialCash}.`);
}

init();