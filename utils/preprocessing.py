""" Preprocessing pipeline. Covers several functions to preprocess the data.
"""
from os import listdir
from os.path import join

import pandas as pd
import pymongo
from tqdm import tqdm


def create_skeleton(folder: str, to_mongoDB: bool = False, lim: int = None, cat: str = None):
    """ Create "real" skeleton from the kinect data.

        The preprocessing mainly covers to convert the x-y-array (pose) in a more readyble form.
        After preprocessing per action, per frame, per skeleton an array with an entry for each
        joint (node) is provide and each joint is represented by an object with "x" and "y"
        coordinates. Finally, the preprocessing step creates the feature matrix.

        Args:
            folder: str
                Folder that contains the json files with the data. The function iterates over each
                file in the folder, hence, it should only contains the proper json files.
            to_mongoDB: bool
                Determines if the result should be pushed to the monogDB.
            lim: None or int
                Limit to restrict the number of files processed from the folder. E.g. for testing.
            cat: None or str
                Category of the data. For instance, "validation". Not required, but recommended when
                writing to the data base, because the data is maintained in the same collection,
                hence you wouldn't be able to determine train and validation set afterwards.
        Return:
            pd.DataFrame: Pandas data frame with the preprocessed data. Outputformat corresponds to
    """

    data = []

    for i, filename in enumerate(tqdm(listdir(folder))):
        if lim is not None and i > lim:
            break

        # Load single instance
        instance = pd.read_json(join(folder, filename))

        # Prepare skeleton data
        skel_data = []

        for k, frame in enumerate(instance["data"]):
            skeletons = []
            skeletons_matrix = []
            for person in frame["skeleton"]:
                skeleton = {}
                feature_matrix = []
                for j in range(0, len(person["pose"]) - 1, 2):
                    skeleton[f"joint{j}"] = {
                        "x": person["pose"][j],
                        "y": person["pose"][j + 1] * -1
                    }
                    feature_matrix.append([person["pose"][j], person["pose"][j + 1] * -1])

                skeletons.append(skeleton)
                skeletons_matrix.append(feature_matrix)

            skel_data.append({
                "frame": k,
                "skeletons": skeletons,
                "feature_matrices": skeletons_matrix
            })

        # Build data frame
        data.append({
            "frames": skel_data,
            "label": instance["label"][0] if len(instance["label"]) > 0 else "",
            "_id": filename,
            "cat": cat
        })

        if to_mongoDB:
            db_url = ""
            # Log in data for mongo DB is read out from a file called .mongoURL
            with open(".mongoURL") as f:
                db_url = f.readlines()

            client = pymongo.MongoClient(db_url)

            collection = client.temporal_info_graph["kinect-skeleton"]
            # collection.insert_one(data[-1])
            collection.update({"_id": data[-1]["_id"]}, data[-1], upsert=True)

    return pd.DataFrame(data)
