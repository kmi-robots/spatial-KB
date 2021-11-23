# Hybrid object recognition
Working repo for hybrid reasoning pipeline (ML-based + knowledge-based) 
which integrates two types of posthoc reasoners:
    - a reasoner which can account for the typical size of objects
    - a reasoner which considers the typical Qualitative Spatial Relations (QSRs) between objects 
        
## Getting started

For cloning the code and switching to the test branch:
```
git clone git@github.com:kmi-robots/spatial-KB.git
cd spatial-KB 
git checkout test
```

Installing Python3 dependencies (tested on Ubuntu 20.04):
```
pip install -r requirements.txt
``` 

You will need PostgreSQL installed, and the PostGIS and SFCGAL extensions enabled.
Useful links (tested on Ubuntu 20.04):
- Install [PostgreSQL 12](https://www.postgresql.org/download/linux/ubuntu/)
- Build [SFCGAL](https://gitlab.com/Oslandia/SFCGAL) from source
- Build [PostGIS](https://postgis.net/install/) from source (with the *--with-sfcgal* flag)
- [PgAdmin 4](https://www.pgadmin.org/download/pgadmin-4-apt/) provides a helpful interface to visualize and query your DB.

Before moving to the next step, create a DB named "gis_database" that you have admin privileges on.
The annotated wall data can be added to the semantic map through the SQL script
available under ```./data/walls.sql```. Be sure to edit the walls.sql file with your SQL username in place of the "postgres" string. 
Then, from terminal, run:
```
psql your_user -d gis_database -f ./data/walls.sql
```


## Data preparation        

You can download the data needed for reproducing our experiments through [this link](https://mega.nz/file/8sMUGJqL#d8FpmTHlQAfEOBoEOhlxrhEkSPUIbDJ7o1fOrUCGs1Q).
All data are in a .zip folder and will need to be extracted and prepared as follows:

```
mv your-path-to/labdata_AAAIMAKE22.zip your-path-to/spatial-KB/data
unzip labdata_AAAIMAKE22.zip
rm labdata_AAAIMAKE22.zip
cd your-path-to/spatial-KB/data 
mv test-imgs/ Labdata/
```

For the raw object-object relationships in Visual Genome (ver 1.4) you will need:
```
cd spatial-KB/data

wget https://visualgenome.org/static/data/dataset/relationships.json.zip
unzip relationships.json.zip
rm relationships.json.zip

```

Prepare size and spatial properties on your local DB for faster inference:

```
cd your-path-to/spatial-KB
python3 ./utils/size_prep.py
python3 ./utils/spatial_prep.py
```


## Sample commands

We provide a command line interface that can be run from terminal.

**Please note that the first run takes much longer because the raw VG data are processed for the first time.** 
Subsequent runs, instead, rely on the pre-processed VG data stored locally.

ML and Size-only ablation:
```
cd your-path-to/spatial-KB
python3 cli.py --rm size
```

To run experiment A:

```
#spatial only ablation
python3 cli.py

#size+spatial waterfall 
python3 cli.py --rm size_spatial --waterfall true

#size+spatial 3 judges
python3 cli.py --rm size_spatial 

```

To run experiment B:

```
#spatial only ablation
python3 cli.py --ql ML

#size+spatial waterfall 
python3 cli.py --rm size_spatial --ql ML --waterfall true

#size+spatial 3 judges
python3 cli.py --rm size_spatial --ql ML
```
