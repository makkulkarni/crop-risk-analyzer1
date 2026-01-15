# crop-risk-analyzer1
Actuarial Crop Risk Analyzer an example framework for assessing the volatility and risk profile of insurance clusters under the PMFBY scheme

🌾 **PMFBY Crop Insurance Portfolio Analyzer**
This application provides a robust framework for assessing the volatility and risk profile of insurance clusters under the PMFBY scheme. By utilizing Monte Carlo simulations, it calculates Expected Burn Rates and "Tail Risk" (Value at Risk) to help insurers and government stakeholders understand potential payout liabilities.

🚀 **The PMFBY Volatility Workflow**
Unlike simple historical averages, this tool models uncertainty by:

Generating Random Shocks: Creating 1,000+ simulated yield scenarios based on historical variability.

Calculating Payouts: Applying Threshold Yields and indemnity levels to every simulation, capped by the cap_default_pct.

Hierarchical Aggregation: Summing currency liabilities from IU → District → Cluster to account for the "Portfolio Effect" (where low losses in one area offset high losses in another).

Statistical Normalization: Converting total payouts back into Burn Rates (Payout / Sum Insured) to determine the probability of extreme loss events.

📂** Input Data Formats**
To run the simulations, the application requires two specific CSV files:

**1. Yield Data File (yield_df)**
This file contains historical yield performance and area coverage. | Column | Description | | :--- | :--- | | State | State name | | Cluster | Assigned PMFBY Cluster | | District | District name | | Tehsil | Administrative sub-division | | IU_id | Insurance Unit identifier (e.g., Village/Gram Panchayat) | | Crop | Name of the crop | | year | Historical year of the record | | Yield_kg_ha | Historical yield in kg per hectare | | area | Area insured (Hectares) |

**2. Threshold Data File (thresh_df)**
This file contains the contractual parameters for the insurance season. | Column | Description | | :--- | :--- | | State | Must match Yield File | | Cluster | Must match Yield File | | District | Must match Yield File | | Crop | Must match Yield File | | Threshold | The guaranteed yield (kg/ha) | | sum_insured_per_ha | Liability per hectare (Rupees) | | cap_default_pct | Maximum payout cap (e.g., 0.25 for 25% of SI) |

**📊 Understanding the Output**
The tool generates a hierarchical summary (IU, District, and Cluster levels) with the following key metrics:

Total SI: The total liability (Area × Sum Insured per Ha).

Burn Rate: The expected payout as a percentage of Total SI (Mean Payout).

Burn Rate SD: The standard deviation of the burn rate. Higher SD = Higher Volatility.

P90 / P95: The "1-in-10" and "1-in-20" year worst-case payout scenarios.

CV (Coefficient of Variation): Calculated as SD / Burn Rate. It represents the risk-to-reward ratio of the cluster.
