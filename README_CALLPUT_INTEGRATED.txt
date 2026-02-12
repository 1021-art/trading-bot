CALL+PUT 통합(SWITCH0) 필수 파일 번들

구성
1) CALL(롱) 싱글스위치(= V42 + C2.3 OR)
 - single_switch_D_RON113_V42v328_ADDboost_relaxed_OUT_FIXED.py
 - V42_ADAPT_v3_28_ADDboost_guarded2_relaxed_FIXED-1.py
 - IDX500_CD12_with_tradelog_CLEAN.py   (C2.3 포함)
 - (실행 결과 테이프) _single_used.csv

2) PUT(숏) 라우터(= CRASH + LITE)
 - short_router_crash_plus_lite_barwise_v3_cd60_SCALE092.py
 - IDX500_SHORT_V52V2_DD07_S055_RISK0130_MTMTRUE.py
 - IDX500_SHORT_LITE_DD07_S055_LRM130_SAFEBOOST112_MTMTRUE.py
 - (실행 결과 테이프) _router_trades.csv

3) CALL/PUT 통합 컨트롤러(SWITCH0, 정밀 MTM 계산 포함)
 - subaccount_switch_controller_CALLSW_PUTrouter_SWITCH0_MTM.py

필수 데이터(CSV, 번들에 미포함)
 - 15m: Binance_BTCUSDT_15m_2021-01-01_to_2025-12-28.csv
 - 1h:  BTCUSDT_PERP_1h_OHLCV_20210101_to_now_UTC.csv

권장 실행 순서
A) CALL 싱글스위치 실행 -> _single_used.csv 생성
B) PUT 라우터 실행 -> _router_trades.csv 생성
C) 통합 컨트롤러 실행 -> CALL+PUT 통합 결과 및 정밀 MTM 산출

주의
 - 시간 기준: UTC
 - SWITCH0: 콜/풋 동시에 보유하지 않음(항상 1개만)
 - peak 계산에서 transfer 제외(컨트롤러 내부 규칙)
