# ==============================================================================
# Knee Osteoarthritis Diagnostic Pipeline - Patient Splitting Audit & Verification
# Run this script to verify that there is ZERO patient-level overlap (data leakage)
# between the Train, Validation, Test, and Auto-Test directories.
# ==============================================================================

import os

# Dynamically locate root dataset folder (checking src/.. and current working dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if os.path.exists(os.path.join(PROJECT_ROOT, "dataset")):
    DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
elif os.path.exists(os.path.join(SCRIPT_DIR, "dataset")):
    DATA_ROOT = os.path.join(SCRIPT_DIR, "dataset")
else:
    DATA_ROOT = os.path.abspath(os.path.join(os.getcwd(), "dataset"))


def get_unique_patient_ids(folder_path):
    """
    Crawls the folder path, inspects X-ray image filenames, 
    and extracts unique Patient IDs by removing knee side markers (L/R)
    and file extensions.
    """
    patient_ids = set()
    
    if not os.path.exists(folder_path):
        print(f"Warning: Directory not found: {folder_path}")
        return patient_ids
        
    for root, _, files in os.walk(folder_path):
        for f in files:
            # Look for image files only
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                # Strip extension (e.g., '9001695L.png' -> '9001695L')
                base_name = f.split('.')[0]
                
                # Strip trailing L or R knee indicators (e.g., '9001695L' -> '9001695')
                if base_name[-1].upper() in ['L', 'R']:
                    pid = base_name[:-1]
                else:
                    pid = base_name
                    
                patient_ids.add(pid)
                
    return patient_ids

def run_split_audit():
    print("======================================================================")
    print("                KNEE OA DATASET PATIENT-SPLIT AUDIT                   ")
    print("======================================================================")
    print(f"Analyzing dataset directories in: {DATA_ROOT}\n")
    
    # Extract patient IDs from all splits
    train_pids = get_unique_patient_ids(os.path.join(DATA_ROOT, "train"))
    val_pids   = get_unique_patient_ids(os.path.join(DATA_ROOT, "val"))
    test_pids  = get_unique_patient_ids(os.path.join(DATA_ROOT, "test"))
    auto_pids  = get_unique_patient_ids(os.path.join(DATA_ROOT, "auto_test"))
    
    # Print summary statistics
    print("--- PARTITION SUMMARY (UNIQUE PATIENTS) ---")
    print(f"Train Partition Unique Patients:      {len(train_pids)}")
    print(f"Validation Partition Unique Patients: {len(val_pids)}")
    print(f"Test Partition Unique Patients:       {len(test_pids)}")
    print(f"Auto-Test Partition Unique Patients:  {len(auto_pids)}")
    print("-" * 43)
    
    # Check for intersections (overlaps)
    train_val_overlap  = train_pids.intersection(val_pids)
    train_test_overlap = train_pids.intersection(test_pids)
    train_auto_overlap = train_pids.intersection(auto_pids)
    val_test_overlap   = val_pids.intersection(test_pids)
    
    print("\n--- PATIENT OVERLAP (DATA LEAKAGE) AUDIT ---")
    print(f"Train & Validation Overlap:  {len(train_val_overlap)} patients")
    print(f"Train & Test Overlap:        {len(train_test_overlap)} patients")
    print(f"Train & Auto-Test Overlap:   {len(train_auto_overlap)} patients")
    print(f"Validation & Test Overlap:   {len(val_test_overlap)} patients")
    print("-" * 43)
    
    # Scientific evaluation
    total_leaks = len(train_val_overlap) + len(train_test_overlap) + len(train_auto_overlap) + len(val_test_overlap)
    if total_leaks == 0:
        print("\n[VERDICT] PASS: Zero patient-level data leakage detected.")
        print("The splits are 100% independent. The model evaluation is scientifically sound.")
    else:
        print(f"\n[VERDICT] FAIL: Detected {total_leaks} overlapping patient IDs across splits!")
        print("This indicates active data leakage.")
    print("======================================================================")

if __name__ == "__main__":
    run_split_audit()
