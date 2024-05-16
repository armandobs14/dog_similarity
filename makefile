

download:
		echo "Downloading images"
		# Download images
		# wget -O data/images/images.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
		wget -O data/images/annotation.tar http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
		wget -O data/images/lists.tar http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
		# Train and test data
		wget -O data/images/train_data.mat http://vision.stanford.edu/aditya86/ImageNetDogs/train_data.mat
		wget -O data/images/test_data.mat http://vision.stanford.edu/aditya86/ImageNetDogs/test_data.mat

clear_dataset:
	rm -f data/tmp/*
	rm -f data/similar_all_images/*
	rm -f data/dissimilar_all_images/*

create_dataset: clear_dataset
	python3 dataset_creation.py

train_test_split:
	python dataset_object_siamese.py