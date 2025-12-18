# TraderAI

TraderAI is a browser-based AI that learns to trade cryptocurrencies in real-time. It uses TensorFlow.js to create a simple reinforcement learning model that evolves as it trades.

## How to Run

This project is designed to run entirely in the browser, with no backend required.

1.  **Navigate to the `/TraderAI` directory in your web browser.**
    *   If you are running a local web server, this would be `http://localhost:PORT/TraderAI/`.
    *   If you are opening the files directly, open `TraderAI/index.html`.

## How It Works

*   **Frontend:** The application is built with plain HTML, CSS, and JavaScript. It uses Chart.js to display the portfolio's value over time.
*   **AI Model:** A simple neural network is created using TensorFlow.js directly in the browser. This model takes the current market data as input and decides whether to "buy" or "sell".
*   **Learning:** The AI learns through reinforcement learning.
    *   **Automatic Rewards:** After each trade, the application waits for 10 minutes and then calculates the profit or loss from that trade. This is used as a reward signal to train the model.
    *   **User Rewards:** You can also provide your own feedback! Next to each trade in the log, there are "+1" and "-1" buttons. Clicking these will also train the model, allowing you to guide its learning.
*   **Data:** The application fetches real-time market data from the Nobitex public API.

**Note on CORS:** This application fetches data directly from a public API. Due to browser security policies (CORS), this might not work if the API provider restricts access. If you open the browser's developer console and see errors related to CORS, it means a simple backend (a proxy) would be required to bypass these restrictions.
