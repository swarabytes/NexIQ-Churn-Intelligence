# Part 9 — OLAP Analytics on Olist Data

## Dataset
- **Orders:** 98,199
- **Customers:** 94,983
- **Date Range:** 2016-09-04 to 2018-09-03
- **States:** 27
- **Categories:** 74

## Fact/Dimension Model
- **Facts:** order_count, customer_count, item_count, total_sales
- **Dimensions:** Time (year, quarter, month), Geography (state, city), Product (category)

## ROLL-UP (Month → Year)
Monthly records rolled up to 3 yearly records.

## DRILL-DOWN (Year → Month)
Selected year 2018 and broke down to 9 monthly records.

## SLICE (customer_state = SP)
41,125 orders from 39,747 customers.

## DICE
Filters: States=[SP,RJ,MG], Years=[2017,2018], Categories=['bed_bath_table', 'health_beauty']
Result: 12,597 orders.

## PIVOT (Rotate)
Rotates multidimensional axes to cross-tabulate dimensions (e.g. State × Year, Category × Quarter) aggregating metrics (total_sales, order_count, customer_count).
Example: State (Rows) × Year (Columns) evaluating total_sales shows SP leading in 2017 (R$ 2.19M) and 2018 (R$ 2.96M).


## Key OLAP Insights
- Year 2018 had the highest order volume (53,531 orders).
- São Paulo (SP) is the most dominant customer state, accounting for 41.9% of all orders.
- The top 2 product categories are: bed_bath_table, health_beauty.
- The Dice result shows concentration of orders in SP/RJ/MG across 2017-2018 for the top categories.
