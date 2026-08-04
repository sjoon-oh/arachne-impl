#!/usr/bin/env gnuplot

# ==============================================================================
# Global arguments and derived settings
# ==============================================================================

# ------------------------------------------------------------------------------
# Input/output arguments
# ------------------------------------------------------------------------------

if (!exists("arg_plotfile"))   arg_plotfile= "plot.gnuplot"

if (!exists("arg_filename_0")) arg_filename_0 = ""
if (!exists("arg_filename_1")) arg_filename_1 = ""
if (!exists("arg_filename_2")) arg_filename_2 = ""
if (!exists("arg_filename_3")) arg_filename_3 = ""
if (!exists("arg_filename_4")) arg_filename_4 = ""
if (!exists("arg_filename_5")) arg_filename_5 = ""
if (!exists("arg_filename_6")) arg_filename_6 = ""
if (!exists("arg_filename_7")) arg_filename_7 = ""
if (!exists("arg_filename_8")) arg_filename_8 = ""
if (!exists("arg_filename_9")) arg_filename_9 = ""

# Common output basename. Do not include an extension.
# Example: arg_export_base = "../out/latency"
if (!exists("arg_export_base")) arg_export_base = "output"
if (!exists("arg_export_name")) arg_export_name = "output"
if (!exists("arg_export_eps"))  arg_export_eps = sprintf("%s/%s.eps", arg_export_base, arg_export_name)

# Both output formats are always generated.
if (!exists("arg_export_png")) arg_export_png = sprintf("%s/%s.png", arg_export_base, arg_export_name)

# ------------------------------------------------------------------------------
# Figure dimensions
# ------------------------------------------------------------------------------

# The figure size is specified only once, using EPS dimensions in inches.
# arg_size_width/arg_size_height are retained as backward-compatible aliases.
if (!exists("arg_size_width"))  arg_size_width  = 6.1
if (!exists("arg_size_height")) arg_size_height = 2.7
if (!exists("arg_eps_width"))   arg_eps_width   = arg_size_width
if (!exists("arg_eps_height"))  arg_eps_height  = arg_size_height

# PNG is derived automatically from the EPS dimensions.
if (!exists("arg_png_dpi")) arg_png_dpi = 450.0

POINTS_PER_INCH = 150

EPS_WIDTH  = arg_eps_width
EPS_HEIGHT = arg_eps_height
PNG_DPI    = arg_png_dpi

# Pixel canvas corresponding to the EPS physical size at PNG_DPI.
PNG_WIDTH  = int(EPS_WIDTH  * PNG_DPI + 0.5)
PNG_HEIGHT = int(EPS_HEIGHT * PNG_DPI + 0.5)

# Scale fonts and line widths so that their physical size matches the EPS output.
PNG_RENDER_SCALE = PNG_DPI / POINTS_PER_INCH

# ------------------------------------------------------------------------------
# Palette/style arguments
# ------------------------------------------------------------------------------

if (!exists("arg_color_palette")) arg_color_palette = "research_muted"

# ------------------------------------------------------------------------------
# Derived variables
# ------------------------------------------------------------------------------

PLOTFILE        = arg_plotfile

FILENAME_0      = arg_filename_0
FILENAME_1      = arg_filename_1
FILENAME_2      = arg_filename_2
FILENAME_3      = arg_filename_3
FILENAME_4      = arg_filename_4
FILENAME_5      = arg_filename_5
FILENAME_6      = arg_filename_6
FILENAME_7      = arg_filename_7
FILENAME_8      = arg_filename_8
FILENAME_9      = arg_filename_9

COLOR_PALETTE   = arg_color_palette

PNG_EXPORT_NAME = arg_export_png
EPS_EXPORT_NAME = arg_export_eps

# ------------------------------------------------------------------------------
# Fonts
# ------------------------------------------------------------------------------

# Font sizes are authored in points using the EPS output as the reference.
# main.gnuplot applies PNG_RENDER_SCALE to the pngcairo terminal automatically.
FONT_GLOBAL     = "Helvetica,24"
FONT_TITLE      = "Helvetica,24"
FONT_XLABEL     = "Helvetica,24"
FONT_YLABEL     = "Helvetica,24"
FONT_XTICS      = "Helvetica,22"
FONT_YTICS      = "Helvetica,22"
FONT_KEY        = "Helvetica,24"

# ------------------------------------------------------------------------------
# Diagnostic output
# ------------------------------------------------------------------------------

if (strlen(FILENAME_0) > 0) print sprintf("Input file 0: %s", FILENAME_0)
if (strlen(FILENAME_1) > 0) print sprintf("Input file 1: %s", FILENAME_1)
if (strlen(FILENAME_2) > 0) print sprintf("Input file 2: %s", FILENAME_2)
if (strlen(FILENAME_3) > 0) print sprintf("Input file 3: %s", FILENAME_3)
if (strlen(FILENAME_4) > 0) print sprintf("Input file 4: %s", FILENAME_4)
if (strlen(FILENAME_5) > 0) print sprintf("Input file 5: %s", FILENAME_5)
if (strlen(FILENAME_6) > 0) print sprintf("Input file 6: %s", FILENAME_6)
if (strlen(FILENAME_7) > 0) print sprintf("Input file 7: %s", FILENAME_7)
if (strlen(FILENAME_8) > 0) print sprintf("Input file 8: %s", FILENAME_8)
if (strlen(FILENAME_9) > 0) print sprintf("Input file 9: %s", FILENAME_9)

print sprintf("EPS output file: %s", EPS_EXPORT_NAME)
print sprintf("EPS size: %.3f x %.3f", EPS_WIDTH, EPS_HEIGHT)
print sprintf("PNG output file: %s", PNG_EXPORT_NAME)
print sprintf("PNG size: %d x %d pixels at %.1f DPI", PNG_WIDTH, PNG_HEIGHT, PNG_DPI)
print sprintf("PNG render scale: %.4f", PNG_RENDER_SCALE)
print sprintf("Color palette: %s", arg_color_palette)
