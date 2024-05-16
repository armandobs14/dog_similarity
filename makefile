

download:
	echo "Downloading images"
	# Download images
	wget -O /tmp/images.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
	wget -O /tmp/annotation.tar http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
	wget -O /tmp/lists.tar http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
	# Train and test data
	# wget -O /tmp/train_data.mat http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat
	# wget -O /tmp/test_data.mat http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat

make_dirs:
	mkdir -p ~/data
	mkdir -p ~/data/Images/tmp
	mkdir -p ~/data/Images/similar_all_images
	mkdir -p ~/data/Images/dissimilar_all_images
uncompress:
	mkdir -p ~/data
	tar -xf /tmp/images.tar -C ~/data/
	tar -xf /tmp/annotation.tar -C ~/data/
	tar -xf /tmp/lists.tar -C ~/data/

clear_dataset:
	rm -f ~/data/tmp/*
	rm -f ~/data/similar_all_images/*
	rm -f ~/data/dissimilar_all_images/*

create_dataset: clear_dataset
	python3 dataset_creation.py

train_test_split:
	python dataset_object_siamese.py