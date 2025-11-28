@echo off
REM PR #1 consolidation -> Creates a private GitHub Gist containing the consolidation for PR #1 and prints the created Gist URL.
REM Usage: set GITHUB_TOKEN=ghp_... then run create_pr1_gist.cmd

SETLOCAL ENABLEDELAYEDEXPANSION

:: Consolidated PR content (Markdown) stored in a temporary file gist_content.md
echo --- > gist_content.md
echo # Relax more_itertools version constraint to resolve dependency conflicts >> gist_content.md
echo >> gist_content.md
echo **Pull Request**: https://github.com/cloudesize67-cmd/jax_privacy/pull/1 >> gist_content.md
echo >> gist_content.md
echo **Summary:** >> gist_content.md
echo The PR relaxes the strict pin for `more_itertools` from `==10.1.0` to `>=10.1.0` in requirements.txt. >> gist_content.md
echo >> gist_content.md
echo **Files changed:** >> gist_content.md
echo - requirements.txt (modified) >> gist_content.md
echo >> gist_content.md
echo **Patch hunk:** >> gist_content.md
echo ```diff >> gist_content.md
echo @@ -14,7 +14,7 @@ >> gist_content.md
echo  jaxlib>=0.7.0 >> gist_content.md
echo  jaxtyping >> gist_content.md
echo  pydantic >> gist_content.md
echo  ml_collections >> gist_content.md
echo -more_itertools==10.1.0 >> gist_content.md
echo +more_itertools>=10.1.0 >> gist_content.md
echo  numpy >> gist_content.md
echo  optax >> gist_content.md
echo  scipy >> gist_content.md
echo ``` >> gist_content.md
echo >> gist_content.md
echo **Merge status:** Merged into `main` (squash), merge commit SHA: 4404c132aa24e161c58cdb9423bf22563ec86865 >> gist_content.md
echo >> gist_content.md
echo **Timestamp:** 2025-11-28 >> gist_content.md
echo >> gist_content.md
echo **Notes:** Relaxing strict pinning reduces dependency conflicts for downstream users. >> gist_content.md
echo --- >> gist_content.md
echo. >> gist_content.md

:: Create the Gist payload JSON file
echo {^>gist_payload.json
echo   "description": "PR #1 consolidation: relax more_itertools pinning",^ >> gist_payload.json
echo   "public": false,^ >> gist_payload.json
echo   "files": {^ >> gist_payload.json
echo     "PR-1-consolidation.md": {^ >> gist_payload.json
echo       "content": "^ >> gist_payload.json
:: Append the contents of gist_content.md into the JSON payload with proper escaping
for /f "usebackq delims=" %%A in ("gist_content.md") do (^ >> gist_payload.json
echo       %%A\n^ >> gist_payload.json
)
echo     }^ >> gist_payload.json
echo   }^ >> gist_payload.json
echo }^ >> gist_payload.json

echo Creating private Gist... 
if "%GITHUB_TOKEN%"=="" (
  echo ERROR: Please set the environment variable GITHUB_TOKEN with a token that has gist scope.
  goto :EOF
)

curl -s -H "Authorization: token %GITHUB_TOKEN%" -H "Content-Type: application/json" -X POST -d @gist_payload.json https://api.github.com/gists > gist_response.json

:: Print result
for /f "usebackq tokens=1* delims=:" %%i in (`type gist_response.json ^| findstr /c:"html_url"`) do (
  set url=%%j
)
if defined url (
  echo Gist created: %url:~2% 
) else (
  echo Failed to create Gist. See gist_response.json for details.
)

echo Cleanup temporary files...
del gist_content.md
del gist_payload.json
echo Done.
ENDLOCAL
