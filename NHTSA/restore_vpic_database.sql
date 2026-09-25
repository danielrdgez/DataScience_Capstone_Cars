/*
Restore the downloaded NHTSA vPIC backup once, without replacing an existing
database. Run from SSMS or sqlcmd connected to the target SQL Server instance.
Files are placed in that instance's default data and log directories. The SQL
Server service account must be able to read the backup file.
*/
IF DB_ID(N'vPICList_Lite') IS NULL
BEGIN
    DECLARE @DataPath nvarchar(4000) =
        CONVERT(nvarchar(4000), SERVERPROPERTY('InstanceDefaultDataPath'));
    DECLARE @LogPath nvarchar(4000) =
        CONVERT(nvarchar(4000), SERVERPROPERTY('InstanceDefaultLogPath'));
    IF @DataPath IS NULL OR @LogPath IS NULL
        THROW 50001, 'SQL Server did not report default data and log paths; restore with SSMS and choose file locations.', 1;
    SET @DataPath = CONCAT(@DataPath,
        CASE WHEN RIGHT(@DataPath, 1) IN (N'\', N'/') THEN N'' ELSE N'\' END);
    SET @LogPath = CONCAT(@LogPath,
        CASE WHEN RIGHT(@LogPath, 1) IN (N'\', N'/') THEN N'' ELSE N'\' END);
    DECLARE @DataFile nvarchar(4000) = CONCAT(@DataPath, N'vPICList_Lite.mdf');
    DECLARE @LogFile nvarchar(4000) = CONCAT(@LogPath, N'vPICList_Lite_log.ldf');

    RESTORE DATABASE [vPICList_Lite]
    FROM DISK = N'E:\DataScience_Capstone_Cars\DATA\vPICList_lite_2026_09.bak'
    WITH FILE = 1,
         MOVE N'vPICList_Lite'
             TO @DataFile,
         MOVE N'vPICList_Lite_log'
             TO @LogFile,
         RECOVERY,
         STATS = 5;
END
ELSE
BEGIN
    PRINT 'Database vPICList_Lite already exists; it was left unchanged.';
END;
