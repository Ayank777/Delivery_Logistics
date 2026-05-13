SELECT
    delivery_partner,
    COUNT(*)                                          AS total_deliveries,
    SUM(predicted_delayed)                            AS predicted_delays,
    ROUND(AVG(predicted_delay_prob) * 100, 2)         AS avg_delay_prob_pct
FROM delivery_predictions
GROUP BY delivery_partner
ORDER BY avg_delay_prob_pct;

-- High-risk deliveries (prob > 0.7)
SELECT *
FROM delivery_predictions
WHERE predicted_delay_prob > 0.70
ORDER BY predicted_delay_prob DESC;

-- partner performance by region
SELECT
    region,
    delivery_partner,
    ROUND(AVG(predicted_delay_prob) * 100, 2) AS avg_delay_pct
FROM delivery_predictions
GROUP BY region, delivery_partner
ORDER BY region, avg_delay_pct;