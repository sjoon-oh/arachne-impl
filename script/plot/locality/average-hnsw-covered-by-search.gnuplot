#!/usr/bin/env gnuplot

# Axes, ticks, border, and grid

# Comma separated
set datafile separator ","

unset grid

set ylabel "Rate (Avg.)" font FONT_YLABEL
set xlabel "efSearch" font FONT_XLABEL

set xtics nomirror font FONT_XTICS
set ytics nomirror font FONT_YTICS

set key \
    top center \
    reverse \
    Left \
    samplen 1 \
    height 0.5 \
    maxrows 1 \
    nobox \
    outside \
    font FONT_KEY

# set lmargin 6
# set rmargin 4
# set bmargin 3

set yrange [0:1]
set xtics nomirror font FONT_XTICS

# Histogram style for clustered bars by alignment category per EFS.
set style data histograms
set style histogram clustered gap 1
set boxwidth 1 relative
set style fill solid 1.0 border -1

set xrange [0.2:3.8]

plot    FILENAME_0 using 2:xtic(1) \
        linecolor rgb COLOR_PALETTE[1] \
        linewidth 3 \
        title "Aligned" with histograms, \
        '' using 3:xtic(1) \
        linecolor rgb COLOR_PALETTE[2] \
        linewidth 3 \
        title "Non-aligned" with histograms, \
        '' using 4:xtic(1) \
        linecolor rgb COLOR_PALETTE[3] \
        linewidth 3 \
        title "Random" with histograms
