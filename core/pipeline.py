from core.agno_workflow import run_agno_pipeline


def run_pipeline(
    symbol="AAPL",
    cash=10000,
    log_path=None,
    start_date=None,
    end_date=None,
    data_provider=None,
):
    return run_agno_pipeline(
        symbol=symbol,
        cash=cash,
        log_path=log_path,
        start_date=start_date,
        end_date=end_date,
        data_provider=data_provider,
    )
