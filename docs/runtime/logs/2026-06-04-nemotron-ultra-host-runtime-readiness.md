# Nemotron Ultra Host Runtime Readiness

status: `READY`
bundle: `/Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`

## Memory
- total: `128.0 GiB`
- free_like: `111.4 GiB`
- active: `10.4 GiB`
- wired: `4.4 GiB`
- compressed: `0.5 GiB`
- memory_pressure: `OK`
- swapins: `6311634`
- swapouts: `6467217`

## Disk
- bundle_volume_free: `1376.0 GiB` at `/System/Volumes/Data`
- log_volume_free: `1376.0 GiB` at `/System/Volumes/Data`

## High RSS Processes
- `0.53 GiB` /opt/homebrew/lib/node_modules/@openai/codex/node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin/bin/codex --yolo resume
- `0.51 GiB` /System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/Metadata.framework/Versions/A/Support/corespotlightd
- `0.37 GiB` /Applications/Discord.app/Contents/Frameworks/Discord Helper (Renderer).app/Contents/MacOS/Discord Helper (Renderer) --type=renderer --user-data-dir=/Users/eric/Library/Application Support/discord --standard-schemes=disclip --secure-schemes=disclip,sentry-ipc --bypasscsp-schemes=sentry-ipc --cors-schemes=sentry-ipc --fetch-schemes=disclip,sentry-ipc --streaming-schemes=disclip --app-path=/Applications/Discord.app/Contents/Resources/app.asar --no-sandbox --no-zygote --enable-blink-features=EnumerateDevices,AudioOutputDevices --autoplay-policy=no-user-gesture-required --lang=en-US --num-raster-threads=4 --enable-zero-copy --enable-gpu-memory-buffer-compositor-resources --enable-main-frame-before-activation --renderer-client-id=6 --time-ticks-at-unix-epoch=-1780706979544332 --launch-time-ticks=1292027421 --shared-files --field-trial-handle=1718379636,r,12230779718568364321,6918296377303196006,262144 --enable-features=ScreenCaptureKitPickerScreen,ScreenCaptureKitStreamPickerSonoma --disable-features=AllowAggressiveThrottlingWithWebSocket,HardwareMediaKeyHandling,IntensiveWakeUpThrottling,MacWebContentsOcclusion,MediaSessionService,ScreenAIOCREnabled,SpareRendererForSitePerProcess,TimeoutHangingVideoCaptureStarts,UseEcoQoSForBackgroundProcess,WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --enable-node-leakage-in-renderers
- `0.36 GiB` /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd
- `0.36 GiB` /Applications/Slack.app/Contents/Frameworks/Slack Helper (Renderer).app/Contents/MacOS/Slack Helper (Renderer) --type=renderer --user-data-dir=/Users/eric/Library/Containers/com.tinyspeck.slackmacgap/Data/Library/Application Support/Slack --standard-schemes=app,slack-webapp-dev --enable-sandbox --secure-schemes=app,slack-webapp-dev,sentry-ipc --bypasscsp-schemes=slack-webapp-dev,sentry-ipc --cors-schemes=slack-webapp-dev,sentry-ipc --fetch-schemes=slack-webapp-dev,sentry-ipc --service-worker-schemes=slack-webapp-dev --app-path=/Applications/Slack.app/Contents/Resources/app-arm64.asar --enable-sandbox --enable-blink-features=ExperimentalJSProfiler --disable-blink-features=CustomizableSelect --force-color-profile=display-p3-d65 --lang=en-US --num-raster-threads=4 --enable-zero-copy --enable-main-frame-before-activation --renderer-client-id=4 --time-ticks-at-unix-epoch=-1780706979542896 --launch-time-ticks=1266432014 --shared-files --field-trial-handle=1718379636,r,3221813994612234732,9347406774625481651,262144 --enable-features=DocumentPolicyIncludeJSCallStacksInCrashReports,PdfUseShowSaveFilePicker,ScreenCaptureKitMac,ScreenCaptureKitPickerScreen,ScreenCaptureKitStreamPickerSonoma --disable-features=AllowAggressiveThrottlingWithWebSocket,CalculateNativeWinOcclusion,DropInputEventsWhilePaintHolding,EnableWatermarkView,EnumerateDevicesRelaxedCache,HardwareMediaKeyHandling,IntensiveWakeUpThrottling,LocalNetworkAccessChecks,LogJsConsoleMessages,PostQuantumKyber,RequestInitiatorSiteLockEnfocement,ScreenAIOCREnabled,SkipEmptyDisplayHotplugEvent,SpareRendererForSitePerProcess,TimeoutHangingVideoCaptureStarts,TraceSiteInstanceGetProcessCreation,WebRtcHideLocalIpsWithMdns,WinRetrieveSuggestionsOnlyOnDemand --variations-seed-version --pseudonymization-salt-handle=1935764596,r,90380828058812681,4875636096419484633,4 --trace-process-track-uuid=3190708990060038890 --window-type=main --seatbelt-client=43
- `0.30 GiB` /System/Library/PrivateFrameworks/PhotoAnalysis.framework/Versions/A/Support/photoanalysisd
- `0.22 GiB` /Applications/Warp.app/Contents/MacOS/stable
- `0.21 GiB` /Applications/Codex.app/Contents/MacOS/Codex

## Warnings
- none

## Interpretation
- Saved speed logs still point at MoE and Mamba compute/dispatch as the primary bottlenecks.
- Host pressure can add noise or stalls, so rerun expensive probes only when this report is READY or warnings are understood.
- Closing high-RSS apps may help stability, but it is not proven to fix the current 65.773 ms MoE and 64.157 ms Mamba buckets.

## Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/host_runtime_readiness.py --bundle /Users/eric/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --log-dir docs/runtime/logs`
