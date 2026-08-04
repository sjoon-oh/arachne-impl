#!/usr/bin/env gnuplot

# Axes, ticks, border, and grid

# Comma separated
set datafile separator ","

unset grid

set ylabel "Rate" font FONT_YLABEL
set xlabel "Steps" font FONT_XLABEL

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

set xrange [0:50]
set xtics 10
set yrange [0:1]

# set boxwidth 1.0 relative
set style fill solid 1.0 border -1

plot    FILENAME_0 \
        using 9 \
        linecolor rgb COLOR_PALETTE[1] \
        linewidth 3 \
        title "Aligned" \
        with lines, \
        FILENAME_1 \
        using 9 \
        linecolor rgb COLOR_PALETTE[2] \
        linewidth 3 \
        title "Non-aligned" \
        with lines, \
        FILENAME_2 \
        using 9 \
        linecolor rgb COLOR_PALETTE[3] \
        linewidth 3 \
        title "Random" \
        with lines
