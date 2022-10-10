#!/bin/sh
set -e
if test "$CONFIGURATION" = "Debug"; then :
  cd /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1
  make -f /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "Release"; then :
  cd /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1
  make -f /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "MinSizeRel"; then :
  cd /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1
  make -f /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1/CMakeScripts/ReRunCMake.make
fi
if test "$CONFIGURATION" = "RelWithDebInfo"; then :
  cd /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1
  make -f /Users/abhishekshivakumar/academics/Iterative-Disparity-Network/build_M1/CMakeScripts/ReRunCMake.make
fi

