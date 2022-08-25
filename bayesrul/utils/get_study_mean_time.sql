/* 
    For the whole study (results/ncmapss/studies/single_obj/studies.db)
    computes the average time of training for one model, for successful trials
*/
SELECT STUDY_ID, avg(TIME) FROM (
    SELECT TRIAL_ID, NUMBER, STUDY_ID, STATE, DATETIME_START, DATETIME_COMPLETE,
        CAST ((
            JulianDay(datetime(DATETIME_COMPLETE)) - JulianDay(datetime(DATETIME_START))
        ) * 24 * 60 * 60 As Integer) / 60.0 AS TIME
    FROM TRIALS 
    WHERE STATE = "COMPLETE"
)
GROUP BY STUDY_ID;
