# Scheduled wrapper for the KR strategy job, invoked by Windows Task
# Scheduler daily after KRX close. Runs the dry-run script (paper trading
# only - this does NOT touch Kiwoom's real/mock API, see kr/live_dry_run_kr.py's
# own header), emails a report on success, and on failure sends a separate
# failure-alert email (since a crash could happen before the script's own
# success-path email code ever runs). Also writes+pushes a heartbeat file
# so the GitHub Actions watchdog (.github/workflows/kr_watchdog.yml) can
# detect a "never ran / never woke up" day, which nothing local can catch.
#
# To switch this to real trading later: change the python invocation below
# from `kr/live_dry_run_kr.py --email` to
# `live/live_kr_trader.py --kiwoom-live --confirm --no-confirm --email`
# - a deliberate, explicit one-line change, not automatic.

$ErrorActionPreference = "Stop"
$RepoRoot = "c:\Users\woosu\Projects\Apps\autotrade"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "kr\scheduled_logs"
$HeartbeatFile = Join-Path $RepoRoot "kr\heartbeat_kr.json"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_$Timestamp.log"
$StdErrFile = Join-Path $LogDir "run_$Timestamp.stderr.log"

Set-Location $RepoRoot

function Write-AndPushHeartbeat {
    param([string]$Status, [string]$ErrorDetail = "")

    $Today = Get-Date -Format "yyyy-MM-dd"
    $Payload = @{
        date      = $Today
        timestamp = (Get-Date).ToString("o")
        status    = $Status
    }
    if ($ErrorDetail) { $Payload["error"] = $ErrorDetail }
    Set-Content -Path $HeartbeatFile -Value ($Payload | ConvertTo-Json) -Encoding utf8

    git add $HeartbeatFile
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        # Nothing changed (e.g. a second run today after an earlier
        # success) - nothing to commit/push, not an error.
        return
    }
    git commit -m "KR scheduled job heartbeat $Today ($Status) [skip ci]" | Out-Null

    # --autostash handles the case where OTHER unrelated uncommitted work
    # happens to be sitting in the tree at run time (stashes it, rebases,
    # pops it back) - this script must never be the reason unrelated WIP
    # gets lost. origin/main accumulates new commits daily from the US-side
    # cron jobs, so a plain `git push` here would otherwise be rejected as
    # non-fast-forward almost every day (confirmed happened on first test).
    # No `2>&1` / pipe here on purpose - redirecting a native command's
    # stderr through the pipeline is a documented PowerShell 5.1 trap
    # (git's own normal "From <url>" fetch progress line goes to stderr
    # and gets wrapped as a NativeCommandError, corrupting $LASTEXITCODE
    # even on a real success - confirmed happening here on the first two
    # attempts, each time on a pull/push that had actually succeeded).
    # Checking $LASTEXITCODE right after the bare command is the reliable
    # way to read a native command's real exit code in this PowerShell version.
    git pull --rebase --autostash origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git pull --rebase failed while pushing the heartbeat - manual intervention needed, check for a real conflict."
    }
    git push
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed after rebase - heartbeat commit exists locally but did not reach origin."
    }
}

try {
    # Run via Start-Process with explicit stdout/stderr file redirection -
    # NOT PowerShell's `*>`/`2>&1` operators, which re-encode captured
    # output as UTF-16LE in Windows PowerShell 5.1 and produced a garbled
    # (every-other-byte-null) log the first time this was tested. Setting
    # PYTHONUTF8/PYTHONIOENCODING makes Python's own stdout UTF-8, and
    # Start-Process writes that stream to the file as raw bytes, untouched.
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $Proc = Start-Process -FilePath $Python `
        -ArgumentList "kr\live_dry_run_kr.py", "--email" `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError $StdErrFile `
        -NoNewWindow -Wait -PassThru
    $ExitCode = $Proc.ExitCode

    if (Test-Path $StdErrFile) {
        $StdErrContent = Get-Content $StdErrFile -Raw
        if ($StdErrContent) {
            Add-Content -Path $LogFile -Value "`n--- stderr ---`n$StdErrContent" -Encoding utf8
        }
    }

    if ($ExitCode -ne 0) {
        throw "kr/live_dry_run_kr.py exited with code $ExitCode"
    }

    # Success: write heartbeat and push it, so the GH Actions watchdog
    # sees today's run happened even if it never needs to alert.
    Write-AndPushHeartbeat -Status "success"
    Write-Output "KR scheduled job succeeded, heartbeat pushed."
}
catch {
    $ErrorMessage = $_.Exception.Message
    $LogTail = ""
    if (Test-Path $LogFile) {
        $LogTail = (Get-Content $LogFile -Tail 40) -join "`n"
    }
    $AlertMessage = "KR scheduled job FAILED.`n`nError: $ErrorMessage`n`nLast 40 lines of log ($LogFile):`n$LogTail"

    & $Python "kr\send_failure_alert.py" "KR scheduled job FAILED" $AlertMessage

    # Still write a heartbeat marked as failed, so the file's presence
    # doesn't silently imply success if something later reads it loosely -
    # but the watchdog checks for status=="success" specifically, not just
    # the date, so a failed run still triggers the watchdog's own alert too
    # (defense in depth: this machine's email AND the GH Actions watchdog).
    # Wrapped in its own try so a push failure here doesn't mask the
    # original failure that's already been emailed above.
    try {
        Write-AndPushHeartbeat -Status "failed" -ErrorDetail $ErrorMessage
    } catch {
        Write-Output "Additionally failed to push the failure heartbeat: $($_.Exception.Message)"
    }

    exit 1
}
