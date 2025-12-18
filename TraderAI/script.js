// TraderAI - Main script
console.log("TraderAI script loaded.");

// --- CONFIG ---
const SYMBOLS = ['BTCIRT', 'ETHIRT', 'USDTIRT'];
const LOOP_INTERVAL = 15000; // 15 seconds
const REWARD_DELAY = 10 * 60 * 1000; // 10 minutes in milliseconds
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
const pendingRewards = []; // To track trades for reward calculation
const portfolioHistory = []; // To track portfolio value over time

// --- API & TRADING LOGIC ---

/**
 * Fetches market data for a given symbol.
 * NOTE: This might fail due to CORS policy if the API doesn't allow browser-based requests.
 * A backend proxy would be the solution if that's the case.
 * @param {string} symbol - The market symbol (e.g., 'BTCIRT').
 * @returns {Promise<Object|null>} An object with price and recentTrades, or null on failure.
 */
async function fetchMarketData(symbol) {
    try {
        const tradesResponse = await fetch(`https://apiv2.nobitex.ir/v2/trades/${symbol}`);
        if (!tradesResponse.ok) {
            throw new Error(`Failed to fetch trades for ${symbol}`);
        }
        const tradesData = await tradesResponse.json();
        const recentTrades = tradesData.trades.slice(0, 20);

        // Nobitex uses different keys for currency pairs in the stats endpoint.
        // We need to parse the symbol to get src and dst currencies.
        let srcCurrency, dstCurrency;
        // This is a simple parser, might need to be more robust for all symbols.
        if (symbol.endsWith('IRT')) {
            dstCurrency = 'rls';
            srcCurrency = symbol.slice(0, -3).toLowerCase();
        } else if (symbol.endsWith('USDT')) {
            dstCurrency = 'usdt';
            srcCurrency = symbol.slice(0, -4).toLowerCase();
        } else {
            console.error(`Unsupported symbol format: ${symbol}`);
            return null;
        }

        const statsResponse = await fetch(`https://apiv2.nobitex.ir/market/stats?srcCurrency=${srcCurrency}&dstCurrency=${dstCurrency}`);
        if (!statsResponse.ok) {
            throw new Error(`Failed to fetch stats for ${symbol}`);
        }
        const statsData = await statsResponse.json();
        const marketStats = statsData.stats[`${srcCurrency}-${dstCurrency}`];
        if (!marketStats) {
             throw new Error(`No stats found for ${srcCurrency}-${dstCurrency}`);
        }
        const price = parseFloat(marketStats.latest);


        return { symbol, price, recentTrades };
    } catch (error) {
        console.error(`Error fetching data for ${symbol}:`, error);
        // A common issue is a CORS error. The browser console will show a more specific message.
        // If that happens, a backend proxy is required to bypass the browser's security restrictions.
        if (error instanceof TypeError) { // Often indicates a network error like CORS
            alert("Could not fetch data from Nobitex API. This is likely due to CORS policy. A backend proxy is needed to run this application.");
            stop(); // Stop the loop if we can't fetch data
        }
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
        pendingRewards.push(tradeInfo);
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
        pendingRewards.push(tradeInfo);
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
    // State: [currentPrice, cash, last20TradeTypes...] (22 total)
    newModel.add(tf.layers.dense({ inputShape: [22], units: 32, activation: 'relu' }));
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
function preprocessData(marketData) {
    if (!marketData || !marketData.recentTrades || marketData.recentTrades.length === 0) {
        return null;
    }

    const { price, recentTrades } = marketData;

    const avgPrice = recentTrades.reduce((acc, trade) => acc + parseFloat(trade.price), 0) / recentTrades.length;
    const normalizedPrice = price / avgPrice;

    const normalizedCash = wallet.cash / 10000; // Based on initial cash

    const tradeTypes = recentTrades.map(trade => (trade.type === 'buy' ? 1 : -1));
    while (tradeTypes.length < 20) {
        tradeTypes.push(0); // Pad if less than 20 trades
    }

    const state = [normalizedPrice, normalizedCash, ...tradeTypes];
    return tf.tensor2d(state, [1, 22]); // Shape: [1, 22]
}

/**
 * Uses the model to make a trading decision using an epsilon-greedy strategy.
 * @param {tf.Tensor} stateTensor - The preprocessed state tensor.
 * @returns {Promise<string>} The decision: 'buy' or 'sell'.
 */
async function makeDecision(stateTensor) {
    if (!model || !stateTensor) {
        return 'hold';
    }

    // Epsilon-greedy strategy for exploration
    if (Math.random() < EXPLORATION_RATE) {
        addLog("Exploring: Making a random decision.");
        return Math.random() < 0.5 ? 'buy' : 'sell';
    }

    // Exploit: Use the model's prediction
    return tf.tidy(() => {
        const prediction = model.predict(stateTensor);
        const actionIndex = prediction.argMax(1).dataSync()[0];
        return actionIndex === 0 ? 'buy' : 'sell';
    });
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

async function checkPendingRewards() {
    const now = Date.now();
    for (let i = pendingRewards.length - 1; i >= 0; i--) {
        const trade = pendingRewards[i];
        if (now - trade.timestamp >= REWARD_DELAY) {
            const currentData = await fetchMarketData(trade.symbol);
            if (currentData) {
                const priceNow = currentData.price;
                const priceThen = trade.price;
                let reward = 0;

                if (trade.action === 'buy') {
                    // Reward is the percentage price increase
                    reward = (priceNow - priceThen) / priceThen;
                } else if (trade.action === 'sell') {
                    // Reward is the percentage price decrease (avoided loss)
                    reward = (priceThen - priceNow) / priceThen;
                }

                await trainModel(trade.stateTensor, trade.action, reward);
                // Remove the trade from pending rewards
                pendingRewards.splice(i, 1);
            }
        }
    }
}


async function runMainLoop() {
    console.log("Running main loop...");
    const tradeAmount = parseFloat(tradeAmountInput.value);
    if (isNaN(tradeAmount) || tradeAmount <= 0) {
        addLog("ERROR: Invalid trade amount.");
        return;
    }

    const marketData = await Promise.all(SYMBOLS.map(fetchMarketData));
    const validMarketData = marketData.filter(d => d !== null);

    for (const data of validMarketData) {
        const stateTensor = preprocessData(data);
        if (stateTensor) {
            const decision = await makeDecision(stateTensor.clone());
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

    await checkPendingRewards();
    updateUI(validMarketData);
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

    updateUI([]);
    addLog("TraderAI initialized. Click 'Start' to begin.");
}

init();