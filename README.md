## Oil Pricing Risk Engine

An institutional-grade, fully automated wholesale fuel purchasing predictor built specifically for independent gas stations and bulk fuel buyers to optimize physical inventory procurement.

This system acts as a headless, serverless data pipeline and machine learning engine. It mathematically correlates the physical supplier rack prices (Graves Oil Company) with the NYMEX commodity futures market to predict whether wholesale gasoline and diesel prices will rise or fall tomorrow, allowing you to buy before hikes and wait before drops.

---

## Model Performance & Historical Edge

Wholesale physical rack pricing is synchronous with futures: Graves Oil sets its rack price at 6:00 PM based on that day's 1:30 PM NYMEX settle. Since the buyer's purchase deadline at the previous day's price is midnight, this creates a physical arbitrage window. The buyer uses NYMEX settlements to predict the upcoming Graves price change, allowing them to order fuel at the old price.

An out-of-sample quantitative audit across three distinct market regimes (2023–2026) reveals the following historical performance envelopes:

### 1. Multi-Year Precision & Savings Baseline

The primary metric of the engine's edge is **expected net savings in cents per gallon (¢/gal)**. Dollar savings are presented as worked examples assuming a single standard 8,500-gallon capacity delivery truck per active alert day.

*   **Unleaded Gasoline (RBOB):**
    *   **Honest Multi-Year Precision Envelope:** **53%–73%** (with an overall historical average of **71.0%** and an average savings of **+6.04¢/gal** per active alert).
    *   **Conservative Floor for Planning:** **53.0%** precision.
    *   **Yearly Out-of-Sample Performance Breakdown (Unleaded):**
        *   **2023 (Moderate Volatility):** 34 alerts | 52.94% precision | **+29.96¢/gal** net savings | **$2,546.60** annual savings.
        *   **2024 (Low-to-Moderate Volatility):** 146 alerts | 54.11% precision | **+61.31¢/gal** net savings | **$5,211.35** annual savings.
        *   **2025 (High Volatility/Low Noise):** 158 alerts | 72.78% precision | **+217.06¢/gal** net savings | **$18,450.10** annual savings.
        *   **Recent Out-of-Sample Window (Late 2025 into 2026):** 80 alerts in 90 days | 96.25% precision | **+542.03¢/gal** savings | **$46,072.55** OOS savings.

*   **Diesel / Heating Oil (HO):**
    *   **Honest Multi-Year Precision Envelope:** **60%–79%** (with an overall historical average of **94.8%** and an average savings of **+9.97¢/gal** per active alert).
    *   **Conservative Floor for Planning:** **60.0%** precision.
    *   **Yearly Out-of-Sample Performance Breakdown (Diesel):**
        *   **2024:** 84 alerts | 59.52% precision | **+69.41¢/gal** net savings | **$5,900.00** annual savings.
        *   **2025:** 178 alerts | 78.65% precision | **+492.13¢/gal** net savings | **$41,831.05** annual savings.

### 2. Payoff Asymmetry & Rockets-and-Feathers Edge

The model's durability across low-precision years (such as 2023 and 2024 at ~53%–59% precision) is driven by **Rockets and Feathers** asymmetric pass-through pricing. Wholesale suppliers raise prices rapidly in response to NYMEX hikes ("rockets") but lower them gradually in response to drops ("feathers"). 

Consequently, the engine's correct alerts capture large moves, while incorrect alerts incur smaller costs. This results in a structurally profitable **Win-to-Loss ratio of 1.06x to 1.50x**:
*   **RBOB (Gasoline) Asymmetry:** Win: 2.75¢ - 5.11¢/gal | Loss: 2.30¢ - 3.87¢/gal (Ratio: 1.17x - 1.32x)
*   **HO (Diesel) Asymmetry:** Win: 3.90¢ - 4.29¢/gal | Loss: 2.87¢ - 3.69¢/gal (Ratio: 1.06x - 1.50x)

> [!CAUTION]
> **Volatility Warning:** If the 2025 precision increase is primarily a result of calm markets rather than model improvement, a return of high-volatility spikes in 2026 could cause performance to revert to the conservative planning floor of **53% (RB)** / **60% (HO)**. Decisions like storage capacity investment should always be stress-tested at the 53% / 60% floors, not the recent 96% regime levels.

---

## Core Features

- **Hourly Ingestion Retries (`ingest_prices.py`)**: Nightly connects via IMAP to read the official supplier invoice. It queries hourly from 8:00 PM to 12:00 AM CT with exponential backoff to prevent false missing-email alarms, handles target date calculations across the midnight boundary, and appends parsed rack prices to an immutable CSV history.
- **Walk-Forward Calibration (`backtest.py`)**: Re-engineers threshold calibration using a robust 3-fold Walk-Forward Validation strategy over the last 365 days of history (90-day out-of-sample test windows). The parameter grid search sweeps training windows $W \in \{120, 180, 240\}$, hike percentiles $Hp \in \{15, 20\}$, and drop percentiles $Dp \in \{80, 85\}$ to maximize **median out-of-sample savings** to prevent backtest overfitting. Statically configured clamping bounds (e.g., `CLAMP_HIKE_MIN: 0.3`, `CLAMP_HIKE_MAX: 3.0` cents, and `CLAMP_DROP_MIN: -3.0`, `CLAMP_DROP_MAX: -0.3` cents) act as emergency guardrails, overriding the percentiles if market volatility collapses or explodes.
- **Contract Roll Day Exclusions**: Excludes anomalous futures price data surrounding CME contract roll days (the 25th of the month, or nearest business day) from both calibration and active trading to prevent false signaling during mechanical liquidity shifts.
- **Dynamic Volatility & Z-Score Conviction**: Evaluates the strength of futures moves using the rolling standard deviation of daily changes ($\sigma$). It translates daily changes into Z-scores to grade alerts by conviction: **High Conviction** ($|Z| \ge 1.5$), **Moderate Conviction** ($1.0 \le |Z| < 1.5$), or **Low Conviction** ($|Z| < 1.0$). Z-score thresholds are smoothed for historical reproducibility.
- **Quantified Deferral Risk (CVaR)**: Computes the 95% Conditional Value-at-Risk (worst-case tail risk) over the optimal window. For "WAIT" alerts, it computes the expected price spike cost:
  *e.g., "Risk Note: On the worst 5% of days historically, rack prices spiked +4.20¢/gal (+$357 per 8,500 gal truck)."*
- **Real-Time SMS & Email Alerts**: Polls the Schwab API and Yahoo Finance API (`RB=F`, `HO=F`, `CL=F`) during CME trading hours. At 2:35 PM CT (post-NYMEX settlement), it sends structured, high-value alerts containing the verdict, Z-score conviction, and tail-risk warnings. It also displays a standard **3:2:1 Crack Spread** priced per barrel of crude: `(2 * RB * 42 + 1 * HO * 42 - 3 * CL) / 3 = 28 * RB + 14 * HO - CL`.
- **Overnight Verification Loop**: Automatically backfills prediction outcomes by comparing them to the next trading day's physical rack price, appending results to `prediction_log.csv` and displaying the confirmation table in morning notifications.
- **Outlook-Safe Weekly Dashboard (`weekly_report.py`)**: Runs every Saturday morning. It calculates cumulative savings, runs a stable 180-day permutation significance test, and formats the dashboard using nested HTML tables for rendering safety.
- **Blockchain-Style Data Validation (`validate_data.py`)**: Protects the database against corruption and manual edits using an append-only registry of SHA-256 hashes (`data/integrity_hashes.csv`), enforcing strict immutability.

---

## System Architecture

1. **Serverless Execution on GitHub Actions**: 100% serverless data pipeline powered by GitHub Actions. Workflows use shared concurrency groups (e.g. `rbob-tracker`) to enforce queueing and prevent `git push` collision errors during parallel operations.
2. **Workflow Topography**: 
   - **Real-Time Tracker (`tracker.yml`)**: Runs every 5 minutes during CME Globex hours to monitor futures and send alerts.
   - **Nightly Ingestion & Backtest (`backtest_ingest.yml`)**: Checks hourly from 8:00 PM to 12:00 AM CT to pull invoices, validate hashes, run walk-forward calibration, and update caches.
   - **Weekly Dashboard (`weekly_report.yml`)**: Generates reports and analytics every Saturday.
   - **CI & Health (`ci.yml`, `heartbeat.yml`, `keepalive.yml`)**: Automated testing, system health notifications, and repository activity maintenance.
3. **Configuration & Data State Storage**: 
   - **`config.json`**: Read-only, statically schema-checked configuration file for core engine variables. Note that under your load-time price lock contract, you pay the price when fuel is physically loaded, not when ordered. To reflect this, personalize `"DISPATCH_SAME_DAY_RATE"` (defaults to `0.50` as a conservative starting point). This is the single most important number to calibrate from your own Bills of Lading before trusting savings estimates in the system.
   - **`metrics_cache.json`**: Dynamically written, ephemeral state cache storing walk-forward thresholds and daily validation metrics without dirtying the static configuration.
   - **`data/*.csv`**: Lightweight, flat-file databases synced directly to the git branch.


---

## Setup Instructions

To deploy this securely to your own private repository:

1. **Fork the Repository**
2. **Configure GitHub Secrets**: Go to **Settings > Secrets and variables > Actions** and add:
   - `GRAVES_EMAIL`: Your corporate Gmail address receiving the Graves invoices.
   - `GRAVES_APP_PASSWORD`: The 16-character Google App Password for that account.
   - `GMAIL_USER`: The email address the bot uses to send the SMS text emails.
   - `GMAIL_APP_PASSWORD`: The Google App Password for the sending account.
   - `TO_EMAIL`: The destination SMS gateway (e.g., `1234567890@vtext.com` for Verizon).
   - `PHONE_SMS_ADDRESS`: Optional comma-separated SMS gateway addresses (falls back to `TO_EMAIL` if empty).
3. **Timezone Verification**: The pipeline strictly relies on the `America/Chicago` timezone for trading hours, execution windows, and date boundaries. 

### Refreshing the Schwab OAuth Token

Run `python3 handshake.py` from the repository. The first run asks for the Schwab app key, app secret, and redirect URI once, then saves only those three values in your macOS Keychain. Future runs load them automatically, so you only authorize in the browser and paste the redirected URL. A successful exchange updates `SCHWAB_REFRESH_TOKEN` in this repository's GitHub Actions secrets automatically through your authenticated GitHub CLI; the token is supplied through standard input and is not printed. Use `python3 handshake.py --configure` to replace saved values or `python3 handshake.py --forget-credentials` to delete them.

---

## Testing & Verification

The repository contains a highly thorough, multi-tiered testing framework:

1. **Comprehensive Test Suite (`comprehensive_test_suite.py`)**: 
   - A unit-test suite with **54 tests** covering all core categories (email parsing, bounds checks, timezone boundaries, lag math, threshold clamping, CVaR calculations, etc.).
2. **Deterministic Day Replay (`replay_day.py`)**:
   - Performs a stateful point-in-time prediction audit on historically logged days to guarantee the system is deterministic, timezone-stable, and free of future-data leakages.
3. **Statistical Validation & Shadow Benchmarks (`verify_statistics.py`)**:
   - Audits model significance against randomized null models (permutation test), evaluates out-of-sample holdout datasets, computes yearly regime shifts, analyzes residual diagnostics, and measures model performance against shadow baselines.
4. **Empirical Audits & Stress Testing (`scratch/`)**:
   - A suite of specialized tools (`conviction_regime_audit.py`, `weekday_performance_audit.py`, `run_simulations.py`, etc.) for identifying boundary conditions, decay curves, weekday characteristics, and risk structures.

---

## Operational Constraints & Safeguards

To maintain pipeline stability and catch gross external data entry errors, the system enforces the following constraints:
1. **$1.00 Daily Price Jump Safeguard**: In `validate_data.py`, physical and settlement price inputs are rejected if they jump by more than $1.00/gal (100 cents/gal) in a single daily transition. In the event of a genuine black swan market shift exceeding this limit, the pipeline will halt as a safety check, requiring administrative override or verification.
2. **$1.00 to $10.00 Price Boundaries**: Absolute price inputs are validated to reside strictly within a [$1.00, $10.00] range.

---

## Disclaimer

This software is a decision-support tool built for informational purposes only. It does not constitute financial advice. The maintainers are not responsible for fuel purchasing decisions, inventory stockouts, or financial losses resulting from the use of this tool.
