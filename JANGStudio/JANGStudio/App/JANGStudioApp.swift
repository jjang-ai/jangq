// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI

@main
struct JANGStudioApp: App {
    @State private var capabilities = CapabilitiesService()
    @State private var profiles = ProfilesService()
    @State private var settings = AppSettings()

    var body: some Scene {
        WindowGroup("JANG Studio") {
            WizardView()
                .frame(minWidth: 960, minHeight: 640)
                .environment(capabilities)
                .environment(profiles)
                .environment(settings)
                .task {
                    await capabilities.refresh()
                    await profiles.refresh()
                }
        }
        .windowResizability(.contentMinSize)

        Settings {
            SettingsWindow(settings: settings)
                .environment(profiles)
                .environment(capabilities)
        }
    }
}
