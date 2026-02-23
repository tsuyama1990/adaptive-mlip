    # Check centering logic:
    # 1. Calculate center of bounding box for new cell:
    #    Center_X = L_x / 2 = 1.5
    #    Center_Y = L_y / 2 = 1.0
    #    Center_Z = L_z / 2 = 1.0
    # 2. Calculate center of original atoms bounding box:
    #    Center_X_orig approx 1000.5
    #    Center_Y_orig approx 1000.0
    #    Center_Z_orig approx 1000.0
    # 3. Calculate shift vector (New_Center - Orig_Center):
    #    Shift_X = 1.5 - 1000.5 = -999.0
    #    Shift_Y = 1.0 - 1000.0 = -999.0
    #    Shift_Z = 1.0 - 1000.0 = -999.0
    # 4. Verify new positions:
    #    Atom 1: [1000, 1000, 1000] + [-999, -999, -999] = [1.0, 1.0, 1.0]
    #    Atom 2: [1001, 1000, 1000] + [-999, -999, -999] = [2.0, 1.0, 1.0]
