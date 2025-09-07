# Aircraft identification database based on SimConnect characteristics
# This helps identify real aircraft models when string variables aren't available

AIRCRAFT_DATABASE = {
    # Format: (engine_type, num_engines, is_retractable, max_weight_range): "Aircraft Name"
    
    # Single Engine Piston Aircraft
    (0, 1, False, (0, 3000)): "Cessna 172 Skyhawk",
    (0, 1, False, (3000, 4000)): "Cessna 182 Skylane", 
    (0, 1, True, (0, 3000)): "Piper Cherokee",
    (0, 1, True, (3000, 4000)): "Beechcraft Bonanza",
    (0, 1, False, (1000, 2000)): "Piper Cub",
    (0, 1, True, (2500, 3500)): "Mooney M20",
    
    # Twin Engine Piston Aircraft  
    (0, 2, True, (4000, 7000)): "Beechcraft Baron",
    (0, 2, True, (5000, 8000)): "Cessna 310",
    (0, 2, True, (6000, 9000)): "Piper Seneca",
    (0, 2, False, (4000, 6000)): "Beechcraft Duchess",
    
    # Single Engine Turboprop
    (2, 1, True, (8000, 15000)): "TBM 930",
    (2, 1, True, (6000, 10000)): "Pilatus PC-12",
    (2, 1, True, (10000, 20000)): "Beechcraft King Air C90",
    
    # Twin Engine Turboprop
    (2, 2, True, (10000, 15000)): "Beechcraft King Air 350",
    (2, 2, True, (15000, 25000)): "Cessna 441 Conquest",
    (2, 2, True, (8000, 12000)): "Piper Cheyenne",
    (2, 2, True, (25000, 40000)): "ATR 42",
    (2, 2, True, (40000, 70000)): "ATR 72",
    
    # Jets - Single Engine
    (1, 1, True, (8000, 15000)): "Cirrus Vision SF50",
    
    # Jets - Twin Engine Light
    (1, 2, True, (8000, 15000)): "Eclipse 500",
    (1, 2, True, (12000, 18000)): "Citation Mustang",
    (1, 2, True, (15000, 25000)): "Phenom 100",
    (1, 2, True, (20000, 30000)): "Citation CJ4",
    
    # Jets - Twin Engine Medium  
    (1, 2, True, (25000, 40000)): "Citation Latitude",
    (1, 2, True, (30000, 45000)): "Learjet 75",
    (1, 2, True, (35000, 50000)): "Gulfstream G280",
    
    # Jets - Twin Engine Heavy
    (1, 2, True, (50000, 80000)): "Gulfstream G650",
    (1, 2, True, (80000, 120000)): "Boeing Business Jet",
    
    # Commercial Aircraft (estimated based on MSFS default aircraft)
    (1, 2, True, (120000, 200000)): "Airbus A320 Neo",
    (1, 2, True, (200000, 300000)): "Boeing 737 MAX",
    (1, 2, True, (300000, 450000)): "Airbus A330",
    (1, 2, True, (450000, 600000)): "Boeing 747-8",
    (1, 2, True, (600000, 800000)): "Airbus A380",
    
    # Helicopters
    (3, 1, None, (0, 5000)): "Robinson R22",
    (3, 1, None, (5000, 10000)): "Bell 206 JetRanger", 
    (3, 2, None, (8000, 15000)): "Bell 412",
    (3, 2, None, (15000, 25000)): "Sikorsky S-76",
}

def identify_aircraft(engine_type, num_engines, is_retractable, max_weight):
    """
    Identify aircraft based on SimConnect characteristics
    """
    if max_weight is None:
        max_weight = 0
        
    # Try exact matches first
    for (e_type, n_engines, retract, weight_range), aircraft_name in AIRCRAFT_DATABASE.items():
        if (e_type == engine_type and 
            n_engines == num_engines and 
            (retract is None or retract == is_retractable) and
            weight_range[0] <= max_weight <= weight_range[1]):
            return aircraft_name
    
    # Try broader matches without weight constraint
    for (e_type, n_engines, retract, weight_range), aircraft_name in AIRCRAFT_DATABASE.items():
        if (e_type == engine_type and 
            n_engines == num_engines and 
            (retract is None or retract == is_retractable)):
            return f"{aircraft_name} (variant)"
    
    # Return None if no match found
    return None