# fiw_family_searcher/data_processing/data_loader.py
import pandas as pd
import os
from glob import glob
from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger

logger = setup_logger(__name__)


def load_rids():
    """Loads relationship IDs and labels."""
    try:
        df = pd.read_csv(config.FIW_RIDS_CSV)
        logger.info(f"Loaded {len(df)} RIDs from {config.FIW_RIDS_CSV}")
        return df
    except FileNotFoundError:
        logger.error(f"FIW_RIDs.csv not found at {config.FIW_RIDS_CSV}")
        return pd.DataFrame()



def load_fids():
    """Loads family IDs and names."""
    try:
        # === THIS IS THE KEY CHANGE ===
        df = pd.read_csv(config.FIW_FIDS_CSV, header=None, names=['FID', 'Label'])
        # ============================
        logger.info(f"Loaded {len(df)} FIDs from {config.FIW_FIDS_CSV} (assigned columns: ['FID', 'Label'])")
        return df
    except FileNotFoundError:
        logger.error(f"FIW_FIDs.csv not found at {config.FIW_FIDS_CSV}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading FIW_FIDs.csv: {e}")
        return pd.DataFrame()

def load_all_family_data():
    """
    Loads all family member data, photo paths, and relationships.
    Returns a list of dictionaries, each representing a person,
    and a list of dictionaries for relationships.
    """
    fids_df = load_fids()
    if fids_df.empty:
        return [], []

    all_persons_data = []
    all_relationships_data = []

    processed_fids_count = 0

    for _, fid_row in fids_df.iterrows():
        fid = fid_row['FID']
        # family_name = fid_row.iloc[1] # Assuming second column is family name string
        # For FIW_FIDs.csv: F0001,abbas.mahmoud - 2nd col is a reference name, not family name per se
        # Let's use FID as family identifier and the string as a representative name
        family_rep_name = fid_row.iloc[1]

        family_folder_path = os.path.join(config.FIDS_FULL_DIR, fid)
        mid_csv_path = os.path.join(family_folder_path, "mid.csv")

        if not os.path.exists(mid_csv_path):
            logger.warning(f"mid.csv not found for FID {fid} at {mid_csv_path}")
            continue

        try:
            members_df = pd.read_csv(mid_csv_path)
        except Exception as e:
            logger.error(f"Error reading mid.csv for FID {fid}: {e}")
            continue

        # Determine relationship columns dynamically
        # Columns like '1', '2', '3', '4' are MIDs of other members
        # The last two columns are 'Name', 'Gender'
        rel_cols_MIDs_str = [col for col in members_df.columns if col.isdigit()]

        family_member_mids = members_df['MID'].tolist()

        for _, member_row in members_df.iterrows():
            mid = member_row['MID']
            person_id = f"{fid}_{mid}"
            person_name = member_row['Name']
            gender = member_row['Gender']

            photo_paths = []
            mid_folder_path = os.path.join(family_folder_path, f"MID{mid}")
            if os.path.isdir(mid_folder_path):
                photo_paths = glob(os.path.join(mid_folder_path, "*.jpg"))
                photo_paths.extend(glob(os.path.join(mid_folder_path, "*.png")))  # Add other formats if needed

            all_persons_data.append({
                "PersonID": person_id,
                "FID": fid,
                "MID": mid,
                "Name": person_name,  # This is the "clean" name from mid.csv
                "Gender": gender,
                "PhotoPaths": photo_paths,
                "FamilyRepName": family_rep_name  # Store the representative name from FIW_FIDs.csv
            })

            # Parse relationships for this person
            for target_mid_str in rel_cols_MIDs_str:
                target_mid = int(target_mid_str)

                # Ensure this target_mid is actually a member of this family
                # (it should be, based on mid.csv structure)
                if target_mid not in family_member_mids:
                    # This might happen if mid.csv has columns for MIDs not listed in MID column
                    # e.g. family of 3, but columns 1,2,3,4
                    logger.debug(
                        f"Target MID {target_mid} from column header not in MID list for family {fid}. Skipping relationship.")
                    continue

                if mid == target_mid:  # Relationship to self
                    continue

                relationship_rid = member_row[target_mid_str]
                if pd.isna(relationship_rid) or int(relationship_rid) == 0:  # 0 is NA/Self in RIDs
                    continue

                person1_id = person_id  # Source
                person2_id = f"{fid}_{target_mid}"  # Target

                all_relationships_data.append({
                    "Person1ID": person1_id,
                    "Person2ID": person2_id,
                    "RID": int(relationship_rid)
                })

        processed_fids_count += 1
        if processed_fids_count % 100 == 0:
            logger.info(f"Processed {processed_fids_count}/{len(fids_df)} FIDs for data loading.")

    logger.info(f"Loaded data for {len(all_persons_data)} persons and {len(all_relationships_data)} relationships.")
    return all_persons_data, all_relationships_data


if __name__ == '__main__':
    logger.info("Testing data loader functions...")
    rids = load_rids()
    logger.info(f"RIDs head:\n{rids.head()}")

    fids = load_fids()
    logger.info(f"FIDs head:\n{fids.head()}")

    # This can be slow if dataset is large
    # persons, relationships = load_all_family_data()
    # logger.info(f"Sample person data: {persons[0] if persons else 'No person data'}")
    # logger.info(f"Sample relationship data: {relationships[0] if relationships else 'No relationship data'}")
    logger.info("Data loader test complete. Full load_all_family_data can be lengthy.")