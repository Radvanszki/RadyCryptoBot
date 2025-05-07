from dotenv import load_dotenv
load_dotenv()  # Betölti a .env fájlt

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import threading
import time
import requests
import hmac
import hashlib
import urllib.parse
import os
import logging
import numpy as np  # Szükséges a volatilitás számításhoz

# Logging konfiguráció
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Környezeti változók betöltése (.env fájlból)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
FEE_RATE = float(os.getenv("FEE_RATE", 0.001))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Global lock a thread-safe módosításokhoz
global_lock = threading.Lock()

# Dash app létrehozása
app = dash.Dash(__name__)
app.title = "RadyBot Dashboard"

# Globális változók
bot_running = False
log_messages = []  # Log üzenetek a UI-n megjelenítéshez
portfolio = {}     # Portfólió: {symbol: {"quantity", "entry", "current", "unrealized", "weight"}}
SIMULATED_CAPITAL = 1000.0  # Ezt a felhasználó adja meg a Live/Pilot fülön (Available Capital)
backtest_results = None    # Backtest eredmények, ha szükséges

def add_log(message: str) -> None:
    """Thread-safe módon hozzáad egy log üzenetet a globális listához, és a konzolra írja."""
    with global_lock:
        log_messages.append(message)
    logger.info(message)

def generate_price_chart() -> dict:
    """Egyszerű demo árdiagram generálása a grafikonhoz."""
    times = pd.date_range(end=pd.Timestamp.now(), periods=20, freq='H')
    prices = [100 + i + (i % 5) * 2 for i in range(20)]
    figure = {
        'data': [go.Scatter(x=times, y=prices, mode='lines+markers', name='BTC/USDT')],
        'layout': go.Layout(title='Árfolyam alakulás', xaxis={'title': 'Idő'}, yaxis={'title': 'Ár (USDT)'})
    }
    return figure

def get_interval_millis(interval: str) -> int:
    """
    Átváltja az intervallumot milliszekundumra.
    Példák: "15m" -> 15*60*1000, "1h" -> 3600*1000, "1d" -> 24*3600*1000.
    """
    if interval.endswith("m"):
        num = int(interval[:-1])
        return num * 60 * 1000
    elif interval.endswith("h"):
        num = int(interval[:-1])
        return num * 3600 * 1000
    elif interval.endswith("d"):
        num = int(interval[:-1])
        return num * 24 * 3600 * 1000
    else:
        return 3600 * 1000  # Alapértelmezett: 1 óra

def calculate_momentum(client, symbol: str, interval: str, periods: int = 4) -> float:
    """
    Számolja a momentumot a megadott periódusok alapján.
    Formula: (utolsó ár / első ár - 1)*100.
    """
    millis = get_interval_millis(interval)
    end_time = int(time.time() * 1000)
    start_time = end_time - periods * millis
    df = client.get_historical_klines(symbol, interval, start_time, end_time)
    if len(df) > 1:
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        return (end_price / start_price - 1) * 100
    return 0.0

def calculate_volatility(client, symbol: str, interval: str, periods: int = 4) -> float:
    """
    Számolja ki a záróárak százalékos hozamának szórását (volatilitás) a megadott periódusban.
    """
    millis = get_interval_millis(interval)
    end_time = int(time.time() * 1000)
    start_time = end_time - periods * millis
    df = client.get_historical_klines(symbol, interval, start_time, end_time)
    if len(df) > 1:
        returns = df['close'].pct_change().dropna()
        return np.std(returns) * 100
    return 0.0

def calculate_risk_adjusted_momentum(client, symbol: str, interval: str, periods: int = 4) -> float:
    """
    Kiszámolja a risk-adjusted momentumot úgy,
    hogy a normál momentumot elosztja a volatilitással (kis epsilon hozzáadásával).
    """
    eps = 1e-6  # Kisebb érték az osztás stabilitásához
    mom = calculate_momentum(client, symbol, interval, periods)
    vol = calculate_volatility(client, symbol, interval, periods)
    return mom / (vol + eps)

def normalize_weights(weights: dict) -> dict:
    """
    Normalizálja a súlyokat úgy, hogy a súlyok összege 1 legyen.
    Csak a pozitív értékeket veszi figyelembe; ha az összeg 0, azonos súlyt rendel.
    """
    positive = {sym: max(score, 0) for sym, score in weights.items()}
    total = sum(positive.values())
    if total == 0:
        n = len(positive)
        return {sym: 1.0/n for sym in positive}
    return {sym: score/total for sym, score in positive.items()}

# Binance API wrapper osztály
class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        """
        Inicializálja a Binance klienst a megadott API kulcsokkal, valamint létrehozza a requests.Session-t.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.binance.com'
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
    
    def _sign_payload(self, params: dict) -> str:
        """Aláírja a paramétereket HMAC-SHA256 segítségével."""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"
    
    def get_historical_klines(self, symbol: str, interval: str, start_str: int, end_str: int = None) -> pd.DataFrame:
        """
        Lekéri a historikus klines adatokat a megadott szimbólumra, és DataFrame-ben adja vissza.
        """
        url = f"{self.base_url}/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'startTime': int(start_str), 'limit': 1000}
        if end_str is not None:
            params['endTime'] = int(end_str)
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()
            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time'] + \
                   [f"extra_{i}" for i in range(7)]
            df = pd.DataFrame(data, columns=cols)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close'] = df['close'].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error in get_historical_klines: {e}")
            return pd.DataFrame()
    
    def get_price_change(self, symbol: str, interval: str) -> (float, float):
        """
        Visszaadja az utolsó két gyertya árváltozását (%) és a legújabb árat.
        """
        start_time = int(time.time() * 1000) - 2 * 60 * 60 * 1000
        df = self.get_historical_klines(symbol, interval, start_time)
        if len(df) >= 2:
            old = df['close'].iloc[-2]
            new = df['close'].iloc[-1]
            return ((new - old) / old) * 100, new
        return None, None
    
    def get_top_gainers(self, interval: str) -> list:
        """
        Lekéri a 24 órás ticker adatokat, majd visszaadja a top 10 USDT páros szimbólumot.
        """
        url = f"{self.base_url}/api/v3/ticker/24hr"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            syms = [item for item in data if item['symbol'].endswith('USDT')]
            sorted_syms = sorted(syms, key=lambda x: float(x['priceChangePercent']), reverse=True)
            return [item['symbol'] for item in sorted_syms[:10]]
        except Exception as e:
            logger.error(f"Error in get_top_gainers: {e}")
            return []
    
    def execute_order(self, symbol: str, side: str, quantity: float, mode: str = 'pilot') -> dict:
        """
        Élő módban végrehajt egy megbízást, egyébként szimulálja.
        """
        if mode == 'live':
            url = f"{self.base_url}/api/v3/order"
            params = {'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': quantity}
            signed_params = self._sign_payload(params)
            try:
                response = self.session.post(url, params=signed_params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'success', 'message': f'Order {side} simulated for {symbol}'}

# Telegram értesítés
def send_telegram(message: str) -> None:
    """
    Ha be vannak állítva a TELEGRAM_TOKEN és TELEGRAM_CHAT_ID változók,
    a megadott üzenettel értesítést küld a Telegramon.
    """
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': f"📢 RadyBot értesítés:\n{message}"}
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            add_log(f"Telegram hiba: {e}")

# Live kereskedési logika

def first_portfolio_load(client: BinanceClient, size: int, interval: str, mode: str) -> None:
    """
    Inicializálja a portfóliót a top gainers listából a risk-adjusted momentum alapján.
    A normalizált súlyok segítségével allokálja az induló kapitalt (SIMULATED_CAPITAL),
    és kiszámolja a vásárlási mennyiséget.
    """
    candidates = client.get_top_gainers(interval)
    risk_scores = {}
    for sym in candidates:
        risk_adj = calculate_risk_adjusted_momentum(client, sym, interval, periods=4)
        risk_scores[sym] = risk_adj
        add_log(f"Risk-adjusted momentum {sym}: {risk_adj:.4f}")
    filtered = {sym: score for sym, score in risk_scores.items() if score > 0}
    if len(filtered) < size:
        chosen = candidates[:size]
        norm = {sym: 1/size for sym in chosen}
    else:
        sorted_syms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:size]
        chosen = [sym for sym, score in sorted_syms]
        norm = normalize_weights({sym: score for sym, score in sorted_syms})
    add_log(f"Választott coinok: {chosen}")
    for sym in chosen:
        _, price = client.get_price_change(sym, interval)
        if price is not None:
            allocated_capital = norm[sym] * SIMULATED_CAPITAL
            quantity = allocated_capital / price * (1 - FEE_RATE)
            with global_lock:
                portfolio[sym] = {
                    "quantity": quantity,
                    "entry": price,
                    "current": price,
                    "unrealized": 0.0,
                    "weight": norm[sym] * 100  # százalékban
                }
            add_log(f"✅ Vásárlás: {sym}, Entry Price: {price:.2f}, Qty: {quantity:.4f}, Súly: {norm[sym]*100:.2f}%")
            send_telegram(f"Vásárlás: {sym}, ár: {price:.2f}, Súly: {norm[sym]*100:.2f}%")

def cycle_logic(client: BinanceClient, interval: str, stop_loss: float, mode: str) -> None:
    """
    Frissíti a portfólió pozícióit:
      - Lekéri az aktuális árat,
      - Számolja az unrealized P/L-t,
      - Amennyiben a stop-loss érték alá esik, eltávolítja a pozíciót.
    """
    with global_lock:
        for sym in list(portfolio.keys()):
            try:
                _, new_price = client.get_price_change(sym, interval)
                if new_price is not None:
                    pos = portfolio[sym]
                    pos['current'] = new_price
                    pos['unrealized'] = ((new_price - pos['entry']) / pos['entry']) * 100
                    if new_price < pos['entry'] * (1 - stop_loss/100):
                        add_log(f"💥 Stop-loss aktiválva {sym}: Current={new_price:.2f}, Entry={pos['entry']:.2f}")
                        send_telegram(f"Stop-loss: {sym}, ár: {new_price:.2f}")
                        del portfolio[sym]
            except Exception as e:
                add_log(f"Error updating {sym}: {e}")

# Backtest logika
def backtest_strategy(client: BinanceClient, symbol: str, interval: str, stop_loss: float, initial_balance: float) -> (pd.DataFrame, float):
    """
    Egyszerű backtest stratégia: ha nincs pozíció, vásárol; ha a stop-loss trigger aktiválódik, elad.
    Visszaadja az ügyleteket tartalmazó DataFrame-et és a végső egyenleget.
    """
    start_time = int(time.time() * 1000) - 24 * 60 * 60 * 1000
    df = client.get_historical_klines(symbol, interval, start_time)
    balance = initial_balance
    position = 0.0
    peak = 0.0
    trades = []
    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        if position > 0:
            peak = max(peak, price)
        else:
            peak = price
        if position == 0:
            position = balance / price * (1 - FEE_RATE)
            balance = 0
            trades.append((df['open_time'].iloc[i], 'BUY', price))
        else:
            if price < peak * (1 - stop_loss/100):
                balance = position * price * (1 - FEE_RATE)
                position = 0
                peak = 0
                trades.append((df['open_time'].iloc[i], 'SELL', price))
    if position > 0:
        balance = position * df['close'].iloc[-1] * (1 - FEE_RATE)
    result_df = pd.DataFrame(trades, columns=['time', 'type', 'price'])
    return result_df, balance

# Dashboard layout, Live/Pilot és Backtest fülök
app.layout = html.Div([
    html.H1('RadyBot Dashboard'),
    dcc.Tabs(id='tabs', value='live', children=[
        dcc.Tab(label='Live/Pilot', value='live'),
        dcc.Tab(label='Backtest', value='backtest')
    ]),
    html.Div(id='tab-content')
])

# Live/Pilot fül tartalma (beépítve az induló kapital input-ot)
live_content = html.Div([
    html.Div([
        html.Label('API Key:'), dcc.Input(id='api-key', type='text')
    ], style={'display': 'inline-block'}),
    html.Div([
        html.Label('API Secret:'), dcc.Input(id='api-secret', type='password')
    ], style={'display': 'inline-block', 'marginLeft': '20px'}),
    html.Br(),
    # Available Capital input mező a Live/Pilot fülön:
    html.Div([
        html.Label('Available Capital:'), dcc.Input(id='initial-capital', type='number', value=1000)
    ], style={'display': 'inline-block', 'marginLeft': '20px'}),
    html.Br(),
    html.Button('Start', id='start-btn'),
    html.Button('Stop', id='stop-btn', style={'marginLeft': '10px'}),
    html.Div(id='status-indicator', style={'marginTop': '10px', 'fontWeight': 'bold'}),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0),
    dcc.Graph(id='price-graph'),
    html.Div(id='log-output', style={'whiteSpace': 'pre-wrap', 'height': '200px', 'overflowY': 'scroll'}),
    html.H2("Portfólió (Súlyozva a risk-adjusted momentum alapján)"),
    html.Div(id='portfolio-table')
])

# Backtest fül tartalma (Initial Balance input már szerepel)
backtest_content = html.Div([
    html.Div([html.Label('Symbol:'), dcc.Input(id='bt-symbol', value='BTCUSDT')]),
    html.Div([html.Label('Interval:'), dcc.Dropdown(id='bt-interval', options=[{'label': i, 'value': i} for i in ['1h','4h','1d']], value='1h')]),
    html.Div([html.Label('Stop-loss %:'), dcc.Input(id='bt-stop', type='number', value=1)]),
    html.Div([html.Label('Initial Balance:'), dcc.Input(id='bt-balance', type='number', value=1000)]),
    html.Button('Run Backtest', id='bt-run'),
    dcc.Graph(id='bt-trades-graph'),
    html.Div(id='bt-result')
])

@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab_value: str):
    """Vált a Live/Pilot és Backtest fülök között."""
    return live_content if tab_value == 'live' else backtest_content

# Live/Pilot mód callback-ja: Frissíti a grafikonokat, logokat és a státusz badge-et
@app.callback(
    [Output('price-graph', 'figure'),
     Output('log-output', 'children'),
     Output('status-indicator', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    [State('api-key', 'value'),
     State('api-secret', 'value'),
     State('initial-capital', 'value')]
)
def update_live(n_intervals, start_btn, stop_btn, api_key, api_secret, available_capital):
    global bot_running, SIMULATED_CAPITAL
    # Beállítjuk az induló kapital értékét az input mezőből
    SIMULATED_CAPITAL = float(available_capital) if available_capital else 1000.0

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if api_key and api_secret:
        client = BinanceClient(api_key, api_secret)
    else:
        add_log("Hiányzó API adatok!")
        return generate_price_chart(), "\n".join(log_messages), html.Span("Stopped", style={'color': 'red', 'fontWeight': 'bold'})
    
    if trigger == 'start-btn':
        bot_running = True
        with global_lock:
            log_messages.clear()
        add_log("Bot elindult")
        if not portfolio:
            first_portfolio_load(client, size=3, interval='1h', mode='pilot')
    elif trigger == 'stop-btn':
        bot_running = False
        with global_lock:
            portfolio.clear()
        add_log("Bot leállt")
    
    if bot_running:
        cycle_logic(client, '1h', stop_loss=1, mode='pilot')
    
    status_badge = (html.Span("Running", style={'color': 'green', 'fontWeight': 'bold'})
                    if bot_running else 
                    html.Span("Stopped", style={'color': 'red', 'fontWeight': 'bold'}))
    
    return generate_price_chart(), "\n".join(log_messages), status_badge

# Callback: Frissíti a portfólió táblázatot
@app.callback(
    Output('portfolio-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_portfolio_table(n_intervals):
    with global_lock:
        if not portfolio:
            return "Nincs portfólió adat."
        header = html.Tr([
            html.Th("Symbol"),
            html.Th("Mennyiség"),
            html.Th("Bekerülési ár"),
            html.Th("Jelenlegi ár"),
            html.Th("Unrealized P/L (%)"),
            html.Th("Súly (%)")
        ], style={'border': '1px solid black', 'padding': '5px'})
        rows = []
        for sym, data in portfolio.items():
            rows.append(html.Tr([
                html.Td(sym, style={'border': '1px solid black', 'padding': '5px'}),
                html.Td(f"{data.get('quantity', 0):.4f}", style={'border': '1px solid black', 'padding': '5px'}),
                html.Td(f"{data.get('entry', 0):.2f}", style={'border': '1px solid black', 'padding': '5px'}),
                html.Td(f"{data.get('current', 0):.2f}", style={'border': '1px solid black', 'padding': '5px'}),
                html.Td(f"{data.get('unrealized', 0):.2f}", style={'border': '1px solid black', 'padding': '5px'}),
                html.Td(f"{data.get('weight', 0):.2f}", style={'border': '1px solid black', 'padding': '5px'})
            ]))
        table = html.Table([header] + rows, style={'width': '100%', 'border-collapse': 'collapse'})
        return table

# Backtest mód callback-ja
@app.callback(
    [Output('bt-trades-graph', 'figure'), Output('bt-result', 'children')],
    Input('bt-run', 'n_clicks'),
    [State('bt-symbol', 'value'),
     State('bt-interval', 'value'),
     State('bt-stop', 'value'),
     State('bt-balance', 'value')]
)
def run_backtest(n_clicks, symbol, interval, stop_loss, initial_balance):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    client = BinanceClient('', '')  # Historikus adatokhoz nem szükséges API kulcs
    trades_df, final_balance = backtest_strategy(client, symbol, interval, stop_loss, initial_balance)
    marker_symbols = [1 if t == 'BUY' else 2 for t in trades_df['type']]
    fig = {
        'data': [go.Scatter(
            x=trades_df['time'],
            y=trades_df['price'],
            mode='markers',
            marker={'size': 10, 'symbol': marker_symbols}
        )],
        'layout': go.Layout(title='Backtest Trades', xaxis={'title': 'Time'}, yaxis={'title': 'Price'})
    }
    result_text = f"Végső egyenleg: {final_balance:.2f}"
    return fig, result_text

if __name__ == '__main__':
    app.run_server(debug=True)
