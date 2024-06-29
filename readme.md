# Bitcoin Trading Bot

This project consists of a Bitcoin Trading BOT that predicts buy or sell signals every 4 hours using Binance data. The bot is triggered every 4 hours, predicts signal for the incoming 4 hour, deploys an order accroding to predefined parameters and informs the user via Telegram. The bot uses a XGBoost model custom tailored for predicting BTC/USDT 4 Hour signal. This model can be developed by model_development file introduced in the project. Project includes a Jupyter notebook for model development, a Python class for the trading bot, several Python functions for technical indicators and necessary features, and a pipeline function to update a local SQLite database with Binance data. For reliable usage model depends on a local sqlite db, which is updated every 4 hour by Binance API. 

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Structure

```
├── .env # Binance Credentials and Telegram Token and Chat id
├── requirements.txt # Requrierements file
├── sqlite.db # Local DB
├── model_development.ipynb # Jupyter notebook for model development
├── features.py # Python functions implementing indicators and features
├── pipeline.py # Pipeline to update local SQLite DB with Binance data
├── predictor.py # Python class for the trading bot
└── run_robot.py # Script to run the trading bot
```

## Features

- **Model Development**: `model_development.ipynb` contains the model development and training process for predicting buy or sell signals.
- **Trading Bot**: `predictor.py` includes a Python class that implements the trading bot which predicts buy or sell signals every 4 hours.
- **Technical Indicators**: `features.py` contains functions to calculate various technical indicators.
- **Data Pipeline**: `pipeline.py` updates a local SQLite database with the latest Binance data.
- **Execution Script**: `run_robot.py` runs the trading bot, making use of the developed model and pipeline.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/bitcoin-trading-bot.git
    cd bitcoin-trading-bot
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your Binance API credentials and Telegram Token and chat_id in an environment file `.env`:
    ```env
    BINANCE_API_KEY=your_api_key
    BINANCE_API_SECRET=your_api_secret
    ```

## Usage

**Run Trading Bot**: Execute the `run_robot.py` script to start the trading bot:
    ```sh
    python run_robot.py
    ```
It updates data and contains each utils to run bot. Pay attention to .env file, it has to be created as explained above.

**Model Development**: Open and work on the `model_development.ipynb` Jupyter notebook to develop and train your model. 

If you train and dump new model you have to change model name in predictor file to use new model.
   
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add your feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.
