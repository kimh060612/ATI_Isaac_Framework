import carb.settings

def configure_isaac_sim_logging():
    carb.settings.get_settings().set("/log/debugConsoleLevel", "Fatal")  # verbose"|"info"|"warning"|"error"|"fatal"
    carb.settings.get_settings().set("/log/enabled", False)
    carb.settings.get_settings().set("/log/outputStreamLevel", "Error")
    carb.settings.get_settings().set("/log/fileLogLevel", "Error")
    carb.settings.get_settings().set("/app/enableDeveloperWarnings", False)
    carb.settings.get_settings().set("/app/scripting/ignoreWarningDialog", True)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/verbose", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/info", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/warning", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/error", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/fatal", False)
