read -p "Enter a data_dir: " data_dir

### to downlaod imdb crop data
wget -O "${data_dir%%/}/data/imdb_crop.tar" https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
tar -xvf "${data_dir%%/}/data/imdb_crop.tar"

### download wiki_drop data

wget -O "${data_dir%%/}/data/wiki_crop.tar" https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
tar -xvf "${data_dir%%/}/data/wiki_crop.tar"

### get shape predictor

wget -O "${data_dir%%/}/models/shape_predictor_68_face_landmarks.dat.bz2" http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d "${data_dir%%/}/models/shape_predictor_68_face_landmarks.dat.bz2"