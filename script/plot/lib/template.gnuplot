#!/usr/bin/env gnuplot

# ==============================================================================
# Entry point
#
# Every invocation generates both files:
#   <arg_export_base>.eps
#   <arg_export_base>.png
#
# Figure dimensions are specified once, in EPS inches. The PNG dimensions and
# rendering scale are derived automatically from the EPS size and PNG_DPI.
# ==============================================================================

# `arg_script_dir` identifies one experiment directory; `arg_lib_dir` is
# this shared locality plotting library.  Both are explicit so invocation is
# independent of the caller's working directory.
if (!exists("arg_script_dir")) arg_script_dir = "."
if (!exists("arg_lib_dir")) arg_lib_dir = "../lib"

LIB_DIR = arg_lib_dir

load sprintf("%s/settings.gnuplot", LIB_DIR)
load sprintf("%s/palette.gnuplot", LIB_DIR)

# ------------------------------------------------------------------------------
# EPS: reference output
# ------------------------------------------------------------------------------

# epscairo and pngcairo share the Cairo/Pango rendering path. This gives more
# consistent font metrics and layout than mixing postscript and pngcairo.
set terminal epscairo \
    enhanced \
    color \
    font FONT_GLOBAL \
    size EPS_WIDTH, EPS_HEIGHT

set output EPS_EXPORT_NAME
load PLOTFILE
unset output

print sprintf("EPS export completed: %s", EPS_EXPORT_NAME)

# ------------------------------------------------------------------------------
# PNG: automatically scaled from the EPS dimensions
# ------------------------------------------------------------------------------

# pngcairo uses a pixel canvas. PNG_WIDTH/PNG_HEIGHT convert the EPS dimensions
# from inches to pixels, while PNG_RENDER_SCALE scales fonts and line widths from
# the 72-point/inch coordinate system to the selected PNG density.
set terminal pngcairo \
    enhanced \
    color \
    font FONT_GLOBAL \
    fontscale PNG_RENDER_SCALE \
    linewidth PNG_RENDER_SCALE \
    pointscale PNG_RENDER_SCALE \
    size PNG_WIDTH, PNG_HEIGHT

set output PNG_EXPORT_NAME
load PLOTFILE
unset output

print sprintf("PNG export completed: %s", PNG_EXPORT_NAME)
