; OAIY's own steps in the Windows installer Tauri builds (tauri.conf.json: bundle > windows > nsis > installerHooks).
;
; An update ends OAIY and starts this installer straight away. A program OAIY ran from its engines folder can still be
; exiting then: the voice server for calls, letting go of two speech models on the GPU, took longer than OAIY waited for
; it, and the installer stopped on "Error opening file for writing ...\resources\engines\oaiy-voice.exe" (Retry then went
; on). OAIY now waits for what it stopped before it starts the installer (update::install in the desktop); this is the same
; wait on the installer's side, for an older OAIY doing the updating and for whatever outlived that wait.
;
; Before any file is written, each file in $INSTDIR\resources\engines is opened for writing, as the installer is about to
; write it; one that cannot be is tried again every half second, 30 seconds in all at most, and then the install goes on
; whatever: a file still held after that gets the installer's own "Error opening file for writing" (Abort, Retry, Ignore),
; as before. Only during an update (/UPDATE, which the updater passes): run by hand while OAIY is open, the installer asks
; to close OAIY right after this hook, and waiting here first would only put that question off. Nothing is ended here: the
; hook only waits.

!include LogicLib.nsh

; Half-second steps: 60 is 30 seconds.
!define OAIY_ENGINES_WAIT_STEPS 60

; Wait until every file in DIR can be opened for writing, or STEPS half-seconds have passed in all (one wait for them all,
; not one each). A folder that is not there is nothing to wait for.
!macro OAIY_AWAIT_WRITABLE DIR STEPS
  Push $0 ; the search
  Push $1 ; a file's name
  Push $2 ; the file, opened
  Push $3 ; half-seconds waited, for all the files
  Push $4 ; this file was said to be waited for
  StrCpy $3 0
  ClearErrors
  FindFirst $0 $1 "${DIR}\*.*"
  ${DoUntil} $1 == ""
    ${If} $1 != "."
    ${AndIf} $1 != ".."
    ${AndIfNot} ${FileExists} "${DIR}\$1\*.*"
      StrCpy $4 0
      ${Do}
        ClearErrors
        FileOpen $2 "${DIR}\$1" a
        ${IfNot} ${Errors}
          FileClose $2
          ${If} $4 = 1
            DetailPrint "$1 has closed"
          ${EndIf}
          ${ExitDo}
        ${EndIf}
        ${If} $3 >= ${STEPS}
          DetailPrint "$1 is still in use: installing all the same"
          ${ExitDo}
        ${EndIf}
        ${If} $4 = 0
          DetailPrint "Waiting for $1 to close..."
          StrCpy $4 1
        ${EndIf}
        Sleep 500
        IntOp $3 $3 + 1
      ${Loop}
    ${EndIf}
    ClearErrors
    FindNext $0 $1
  ${Loop}
  ${If} $0 != ""
    FindClose $0
  ${EndIf}
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Pop $0
!macroend

!macro NSIS_HOOK_PREINSTALL
  ${If} $UpdateMode = 1
    !insertmacro OAIY_AWAIT_WRITABLE "$INSTDIR\resources\engines" ${OAIY_ENGINES_WAIT_STEPS}
  ${EndIf}
!macroend
