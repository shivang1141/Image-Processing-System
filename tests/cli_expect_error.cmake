# Runs the CLI and passes only if it exits non-zero AND stderr matches EXPECT.
# Usage: cmake -DEXE=<imgproc> -DARGS=<arg|arg|...> -DEXPECT=<regex> -P cli_expect_error.cmake
string(REPLACE "|" ";" args "${ARGS}")
execute_process(COMMAND "${EXE}" ${args}
                RESULT_VARIABLE rc
                OUTPUT_VARIABLE out
                ERROR_VARIABLE err)

if(rc EQUAL 0)
    message(FATAL_ERROR "Expected a non-zero exit code, got 0.\nstdout:\n${out}")
endif()
if(NOT err MATCHES "${EXPECT}")
    message(FATAL_ERROR "stderr did not match '${EXPECT}'.\nexit code: ${rc}\nstderr:\n${err}")
endif()
message(STATUS "exit code ${rc}, stderr: ${err}")
