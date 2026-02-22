#!/bin/bash
cd /home/ku_ai1/sulee/졸업논문/data

echo "Extracting 100Hz URLs from ptbxl_database.csv..."
awk -F',' 'NR>1 {
    # remove quotes if any
    gsub(/"/, "", $26);
    print "https://physionet.org/files/ptb-xl/1.0.3/"$26".hea";
    print "https://physionet.org/files/ptb-xl/1.0.3/"$26".dat";
}' ptbxl_database.csv > urls.txt

TOTAL=$(wc -l < urls.txt)
echo "Found $TOTAL files to download."

echo "Downloading in parallel (32 threads)..."
cat urls.txt | xargs -n 1 -P 32 wget -q -c -x -nH --cut-dirs=3 -P /home/ku_ai1/sulee/졸업논문/data/

echo "Done!"
