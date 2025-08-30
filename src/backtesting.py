import pandas as pd
import numpy as np

def simulate_trading(prices, predictions, initial_balance=1000.0, trade_percent=0.1, sl_pct=0.015, tp_pct=0.02): 
    balance = initial_balance 
    trade_active = False 
    entry_price = None 
    direction = None 
    trades = [] 
    equity_curve = [balance]
    for i in range(len(prices)):
        price = prices[i]
        pred = predictions[i]

        if not trade_active:
            entry_price = price
            trade_size = balance * trade_percent
            if pred == 'up':
                direction = 'long'
            elif pred == 'down':
                direction = 'short'
            else:
                equity_curve.append(balance)
                continue
            trade_active = True
            trades.append({
                'Entry Index': i,
                'Entry': entry_price,
                'Direction': direction,
                'Trade Size': trade_size
            })
        else:
            tp = entry_price * (1 + tp_pct) if direction == 'long' else entry_price * (1 - tp_pct)
            sl = entry_price * (1 - sl_pct) if direction == 'long' else entry_price * (1 + sl_pct)
            exit_trade = None

            if direction == 'long':
                if price >= tp:
                    profit = trade_size * tp_pct
                    exit_trade = 'TP'
                elif price <= sl:
                    profit = -trade_size * sl_pct
                    exit_trade = 'SL'
            elif direction == 'short':
                if price <= tp:
                    profit = trade_size * tp_pct
                    exit_trade = 'TP'
                elif price >= sl:
                    profit = -trade_size * sl_pct
                    exit_trade = 'SL'

            if exit_trade:
                trades[-1].update({
                    'Exit Index': i,
                    'Exit': price,
                    'Result': exit_trade,
                    'Profit': profit,
                    'Balance Before': balance,
                    'Balance After': balance + profit
                })
                balance += profit
                trade_active = False

        equity_curve.append(balance)

    trades_df = pd.DataFrame(trades)
    # Metrics
    closed_trades = trades_df.dropna(subset=['Profit'])
    profits = closed_trades['Profit'].values
    wins = profits[profits > 0]
    losses = profits[profits < 0]

    metrics = {}
    metrics['Final Balance'] = balance
    metrics['Net Profit'] = profits.sum()
    metrics['Total Trades'] = len(closed_trades)
    metrics['Wins'] = len(wins)
    metrics['Losses'] = len(losses)
    metrics['Win Rate (%)'] = (len(wins) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
    metrics['Average Profit per Trade'] = profits.mean() if len(profits) > 0 else 0
    metrics['Profit Factor'] = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    metrics['Payoff Ratio'] = wins.mean() / abs(losses.mean()) if len(losses) > 0 else np.inf
    metrics['Win/Loss Ratio'] = len(wins) / len(losses) if len(losses) > 0 else np.inf
    metrics['ROI (%)'] = (balance - initial_balance) / initial_balance * 100

    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    metrics['Max Drawdown (%)'] = drawdown.max() * 100

    returns = np.diff(equity_curve) / equity_curve[:-1]
    metrics['Sharpe Ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for p in profits:
        if p > 0:
            cur_win += 1
            max_win_streak = max(max_win_streak, cur_win)
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            max_loss_streak = max(max_loss_streak, cur_loss)
            cur_win = 0
    metrics['Max Win Streak'] = max_win_streak
    metrics['Max Loss Streak'] = max_loss_streak

    avg_win = wins.mean() if len(wins) > 0 else 0 
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0 
    win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0 
    loss_rate = 1 - win_rate 
    metrics['Expectancy'] = win_rate * avg_win - loss_rate * avg_loss
    if 'Entry Index' in closed_trades and 'Exit Index' in closed_trades: 
        metrics['Average Holding Period'] = (closed_trades['Exit Index'] - closed_trades['Entry Index']).mean() 
    else: 
        metrics['Average Holding Period'] = np.nan

    return trades_df, metrics