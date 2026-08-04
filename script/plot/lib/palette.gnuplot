#!/usr/bin/env gnuplot

# ==============================================================================
# Named color palettes
#
# Select a palette with:
#   -e "arg_color_palette='blue_gold'"
#
# The palettes are curated for multi-series plots. Several sets are adapted from
# color combinations collected at https://colorpalette.org/ and reordered so
# adjacent plot series are easier to distinguish.
#
# Backward-compatible aliases palette1 ... palette5 are retained.
# ==============================================================================

array COLOR_PALETTE[10]

# Muted cool colors; default.
if (arg_color_palette eq "research_muted") {
    COLOR_PALETTE[1] = "#817988"  # muted purple
    COLOR_PALETTE[2] = "#98adc9"  # muted blue
    COLOR_PALETTE[3] = "#595163"  # muted dark purple
    COLOR_PALETTE[4] = "#44667f"  # muted slate blue
    COLOR_PALETTE[5] = "#b2a5b3"  # muted lavender
    COLOR_PALETTE[6] = "#9095b0"  # muted gray-blue
    COLOR_PALETTE[7] = "#30303e"  # muted charcoal
    COLOR_PALETTE[8] = "#053348"  # muted navy
    COLOR_PALETTE[9] = "#34576b"  # muted steel blue
    COLOR_PALETTE[10] = "#ebe1e7"  # muted light
}
else if (arg_color_palette eq "blue_gold") {
    # Blue, light blue, gold, green, brown, and neutral accents.
    COLOR_PALETTE[1] = "#2b668b"  # blue
    COLOR_PALETTE[2] = "#c7a04d"  # gold
    COLOR_PALETTE[3] = "#64a3cd"  # light blue
    COLOR_PALETTE[4] = "#719095"  # muted green
    COLOR_PALETTE[5] = "#69501f"  # brown
    COLOR_PALETTE[6] = "#d0af87"  # light brown
    COLOR_PALETTE[7] = "#222513"  # dark green
    COLOR_PALETTE[8] = "#696f26"  # olive
    COLOR_PALETTE[9] = "#98adc1"  # light gray-blue
    COLOR_PALETTE[10] = "#ced6de"  # light gray
}
else if (arg_color_palette eq "slate_blue") {
    # Cool slate and powder-blue palette.
    COLOR_PALETTE[1] = "#546a84"  # slate
    COLOR_PALETTE[2] = "#a6a27c"  # powder blue
    COLOR_PALETTE[3] = "#7698af"  # light slate
    COLOR_PALETTE[4] = "#404c62"  # dark slate
    COLOR_PALETTE[5] = "#7c7d5d"  # muted green
    COLOR_PALETTE[6] = "#7aa4be"  # light blue
    COLOR_PALETTE[7] = "#a29d68"  # muted yellow
    COLOR_PALETTE[8] = "#a0c2d4"  # light gray-blue
    COLOR_PALETTE[9] = "#cce6f1"  # light blue
    COLOR_PALETTE[10] = "#e9f4f7"  # very light blue
}
else if (arg_color_palette eq "earth_teal") {
    # Earth, teal, orange, and gold.
    COLOR_PALETTE[1] = "#32977d"  # teal
    COLOR_PALETTE[2] = "#eb8b1e"  # orange
    COLOR_PALETTE[3] = "#7b3e12"  # brown
    COLOR_PALETTE[4] = "#5084a8"  # slate blue
    COLOR_PALETTE[5] = "#886039"  # muted gold
    COLOR_PALETTE[6] = "#fbce4a"  # gold
    COLOR_PALETTE[7] = "#3f2210"  # dark brown
    COLOR_PALETTE[8] = "#e5a160"  # light brown
    COLOR_PALETTE[9] = "#bebbc4"  # light gray
    COLOR_PALETTE[10] = "#e1b999"  # light tan
}
else if (arg_color_palette eq "deep_blue") {
    # Deep blue with magenta and olive accents.
    COLOR_PALETTE[1] = "#2555c7"  # deep blue
    COLOR_PALETTE[2] = "#a93965"  # magenta
    COLOR_PALETTE[3] = "#51a3dc"  # light blue
    COLOR_PALETTE[4] = "#c1a745"  # gold
    COLOR_PALETTE[5] = "#263f83"  # muted blue
    COLOR_PALETTE[6] = "#574655"  # muted purple
    COLOR_PALETTE[7] = "#0b1a5b"  # dark blue
    COLOR_PALETTE[8] = "#998ba7"  # muted lavender
    COLOR_PALETTE[9] = "#04040b"  # very dark
    COLOR_PALETTE[10] = "#c5e3ee"  # light
}
else if (arg_color_palette eq "orange_navy") {
    # Orange, navy, steel blue, and warm neutrals.
    COLOR_PALETTE[1] = "#f58d33"  # orange
    COLOR_PALETTE[2] = "#3d5565"  # navy
    COLOR_PALETTE[3] = "#d27f67"  # steel blue
    COLOR_PALETTE[4] = "#041a3f"  # dark blue
    COLOR_PALETTE[5] = "#ee9e59"  # light orange
    COLOR_PALETTE[6] = "#6c8192"  # light gray-blue
    COLOR_PALETTE[7] = "#5d4e36"  # dark brown
    COLOR_PALETTE[8] = "#e6ad73"  # light brown
    COLOR_PALETTE[9] = "#2f2418"  # dark gray
    COLOR_PALETTE[10] = "#d2ba9d"  # light tan
}
else if (arg_color_palette eq "forest_cottage") {
    # Forest/cottage colors adapted from a colorpalette.org collection.
    COLOR_PALETTE[1] = "#593029"  # dark brown
    COLOR_PALETTE[2] = "#b7cb62"  # light green
    COLOR_PALETTE[3] = "#8e665c"  # muted brown
    COLOR_PALETTE[4] = "#525632"  # dark green
    COLOR_PALETTE[5] = "#e3c08f"  # light tan
    COLOR_PALETTE[6] = "#8b8a10"  # olive
    COLOR_PALETTE[7] = "#ba836a"  # light brown
    COLOR_PALETTE[8] = "#b5a763"  # muted yellow
    COLOR_PALETTE[9] = "#b4a39b"  # light gray
    COLOR_PALETTE[10] = "#e3dad3"  # light pink
}
else if (arg_color_palette eq "ocean_mist") {
    # Ocean and mist colors.
    COLOR_PALETTE[1] = "#0b3954"  # dark blue
    COLOR_PALETTE[2] = "#087e8b"  # teal
    COLOR_PALETTE[3] = "#ff5a5f"  # coral
    COLOR_PALETTE[4] = "#5c9ead"  # ocean blue
    COLOR_PALETTE[5] = "#334e68"  # navy
    COLOR_PALETTE[6] = "#8ab6d6"  # light blue
    COLOR_PALETTE[7] = "#243b53"  # dark gray-blue
    COLOR_PALETTE[8] = "#9fb3c8"  # light gray-blue
    COLOR_PALETTE[9] = "#bfd7ea"  # very light blue
    COLOR_PALETTE[10] = "#d9e2ec"  # light gray
}
else if (arg_color_palette eq "sunset") {
    # Warm sunset colors.
    COLOR_PALETTE[1] = "#7a1f5b"  # deep red
    COLOR_PALETTE[2] = "#f95738"  # orange
    COLOR_PALETTE[3] = "#edae49"  # golden
    COLOR_PALETTE[4] = "#2e294e"  # deep blue
    COLOR_PALETTE[5] = "#d1495b"  # rose
    COLOR_PALETTE[6] = "#ee964b"  # light orange
    COLOR_PALETTE[7] = "#5d2e46"  # dark purple
    COLOR_PALETTE[8] = "#f4d35e"  # yellow
    COLOR_PALETTE[9] = "#a63d40"  # burgundy
    COLOR_PALETTE[10] = "#c97c5d"  # tan
}
else if (arg_color_palette eq "pastel") {
    # Soft pastel colors.
    COLOR_PALETTE[1] = "#6d8ea0"  # soft blue
    COLOR_PALETTE[2] = "#f2b5d4"  # soft pink
    COLOR_PALETTE[3] = "#9bbf8a"  # soft green
    COLOR_PALETTE[4] = "#c7a6d8"  # soft purple
    COLOR_PALETTE[5] = "#d99b73"  # soft orange
    COLOR_PALETTE[6] = "#8fb9aa"  # soft teal
    COLOR_PALETTE[7] = "#b6d8f2"  # soft light blue
    COLOR_PALETTE[8] = "#f7d6e0"  # soft light pink
    COLOR_PALETTE[9] = "#c7ceea"  # soft light purple
    COLOR_PALETTE[10] = "#ffdac1"  # soft light orange
}
else if (arg_color_palette eq "high_contrast") {
    # High-contrast categorical colors, suitable when series separation matters.
    COLOR_PALETTE[1] = "#0072b2"  # blue
    COLOR_PALETTE[2] = "#d55e00"  # orange
    COLOR_PALETTE[3] = "#009e73"  # green
    COLOR_PALETTE[4] = "#cc79a7"  # magenta
    COLOR_PALETTE[5] = "#e69f00"  # yellow
    COLOR_PALETTE[6] = "#56b4e9"  # light blue
    COLOR_PALETTE[7] = "#332288"  # dark blue
    COLOR_PALETTE[8] = "#000000"  # black
    COLOR_PALETTE[9] = "#999999"  # gray
    COLOR_PALETTE[10] = "#f0e442"  # light yellow
}
else if (arg_color_palette eq "mono_blue") {
    # Sequential blue shades.
    COLOR_PALETTE[1] = "#08306b"  # dark blue
    COLOR_PALETTE[2] = "#08519c"  # medium blue
    COLOR_PALETTE[3] = "#2171b5"  # light blue
    COLOR_PALETTE[4] = "#4292c6"  # lighter blue
    COLOR_PALETTE[5] = "#6baed6"  # lightest blue
    COLOR_PALETTE[6] = "#9ecae1"  # very light blue
    COLOR_PALETTE[7] = "#c6dbef"  # extremely light blue
    COLOR_PALETTE[8] = "#deebf7"  # almost white blue
    COLOR_PALETTE[9] = "#eff3ff"  # very almost white blue
    COLOR_PALETTE[10] = "#f7fbff"  # white blue
}
else if (arg_color_palette eq "terrace_field") {
    # Terrace Field Hill Station, colorpalette.org:
    # https://colorpalette.org/terrace-field-hill-station-color-palette/
    # Agricultural greens with yellow and cool-water accents.
    COLOR_PALETTE[1] = "#4a532d"  # dark olive green
    COLOR_PALETTE[2] = "#629893"  # sea green
    COLOR_PALETTE[3] = "#d5cf39"  # yellow green
    COLOR_PALETTE[4] = "#6c762f"  # olive
    COLOR_PALETTE[5] = "#7ebfb1"  # turquoise
    COLOR_PALETTE[6] = "#a9a831"  # muted yellow
    COLOR_PALETTE[7] = "#202719"  # dark forest
    COLOR_PALETTE[8] = "#7ba59a"  # muted teal
    COLOR_PALETTE[9] = "#99d2bf"  # pale teal
    COLOR_PALETTE[10] = "#d9f3d8"  # pale green
}
else if (arg_color_palette eq "cityscape") {
    # Metropolitan Area Cityscape City, colorpalette.org:
    # https://colorpalette.org/metropolitan-area-cityscape-city-color-palette-40/
    # Dark teal foundations with cyan and red accents.
    COLOR_PALETTE[1] = "#0b5b6b"  # deep teal
    COLOR_PALETTE[2] = "#bb0f26"  # crimson
    COLOR_PALETTE[3] = "#06aab9"  # bright cyan
    COLOR_PALETTE[4] = "#d15a3c"  # warm orange-red
    COLOR_PALETTE[5] = "#05303d"  # dark blue-teal
    COLOR_PALETTE[6] = "#61bec3"  # light turquoise
    COLOR_PALETTE[7] = "#571524"  # dark claret
    COLOR_PALETTE[8] = "#ae6c7d"  # muted rose
    COLOR_PALETTE[9] = "#044255"  # navy
    COLOR_PALETTE[10] = "#d3d5d7"  # cool gray
}
else if (arg_color_palette eq "winter_sky") {
    # Winter Sky Nature, colorpalette.org:
    # https://colorpalette.org/winter-sky-nature-color-palette/
    # A blue-forward sequential palette with enough contrast for line charts.
    COLOR_PALETTE[1] = "#0a4298"  # royal blue
    COLOR_PALETTE[2] = "#077ec7"  # clear blue
    COLOR_PALETTE[3] = "#3c6291"  # steel blue
    COLOR_PALETTE[4] = "#318fce"  # sky blue
    COLOR_PALETTE[5] = "#2b4269"  # dark slate blue
    COLOR_PALETTE[6] = "#4c95cd"  # medium blue
    COLOR_PALETTE[7] = "#689fcd"  # muted blue
    COLOR_PALETTE[8] = "#91b8dc"  # powder blue
    COLOR_PALETTE[9] = "#a7cce8"  # pale blue
    COLOR_PALETTE[10] = "#d2e3f0"  # ice blue
}
else if (arg_color_palette eq "woodland_autumn") {
    # Nature Woodland Autumn, colorpalette.org:
    # https://colorpalette.org/nature-woodland-autumn-color-palette/
    # High-contrast burnt orange/brown accents over warm neutrals.
    COLOR_PALETTE[1] = "#ba3d20"  # burnt red
    COLOR_PALETTE[2] = "#d37b2b"  # autumn orange
    COLOR_PALETTE[3] = "#5d513c"  # olive brown
    COLOR_PALETTE[4] = "#823324"  # dark rust
    COLOR_PALETTE[5] = "#532220"  # deep brown
    COLOR_PALETTE[6] = "#d79655"  # tan orange
    COLOR_PALETTE[7] = "#9d9071"  # muted khaki
    COLOR_PALETTE[8] = "#8e7155"  # brown gray
    COLOR_PALETTE[9] = "#bc8e7e"  # dusty rose
    COLOR_PALETTE[10] = "#f7ead6"  # warm off-white
}
else if (arg_color_palette eq "grayscale") {
    # Monochrome output or print-friendly fallback.
    COLOR_PALETTE[1] = "#111111"  # very dark gray
    COLOR_PALETTE[2] = "#333333"  # dark gray
    COLOR_PALETTE[3] = "#555555"  # medium dark gray
    COLOR_PALETTE[4] = "#777777"  # medium gray
    COLOR_PALETTE[5] = "#999999"  # medium light gray
    COLOR_PALETTE[6] = "#aaaaaa"  # light gray
    COLOR_PALETTE[7] = "#bbbbbb"  # very light gray
    COLOR_PALETTE[8] = "#cccccc"  # extremely light gray
    COLOR_PALETTE[9] = "#dddddd"  # almost white gray
    COLOR_PALETTE[10] = "#eeeeee"  # white gray
}
else {
    # Unknown palette name: use the default rather than leaving the array empty.
    print sprintf("Unknown palette '%s'; using research_muted", arg_color_palette)
    COLOR_PALETTE[1] = "#817988"
    COLOR_PALETTE[2] = "#98adc9"
    COLOR_PALETTE[3] = "#595163"
    COLOR_PALETTE[4] = "#44667f"
    COLOR_PALETTE[5] = "#b2a5b3"
    COLOR_PALETTE[6] = "#9095b0"
    COLOR_PALETTE[7] = "#30303e"
    COLOR_PALETTE[8] = "#053348"
    COLOR_PALETTE[9] = "#34576b"
    COLOR_PALETTE[10] = "#ebe1e7"
}
