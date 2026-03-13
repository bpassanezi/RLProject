import random

def generate_randomized_routes(filepath="nets/random_routes.rou.xml", max_steps=3600):
    """
    Generates a SUMO route file with a baseline steady flow and a randomized 
    traffic spike (rush hour) to prevent the RL agent from overfitting.
    """
    
    # 1. Randomize the Spike Parameters every episode
    spike_start = random.randint(400, 1500)       # Spike starts anywhere between 400s and 1500s
    spike_duration = random.randint(600, 1200)    # Lasts between 10 to 20 minutes
    spike_size = random.randint(500, 2000)
    spike_end = min(spike_start + spike_duration, max_steps)
    
    # Randomize which direction gets hit with the heavy traffic
    spike_axis = random.choice(["EW", "NS"])      
    
    # Randomize the intensity of the spike
    # Probability of a car spawning per second (0.4 to 0.7 is very heavy traffic)
    spike_prob = round(random.uniform(0.5, 0.9), 2) 

    # Baseline traffic probability (steady, light traffic)
    base_prob = 0.1 

    # 2. Write the XML file
    with open(filepath, "w") as routes:
        routes.write('<routes>\n')
        
        # Define the standard vehicle type
        routes.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>\n\n')
        
        # Define the straight routes (using your edge names: incoming -> outgoing)
        routes.write('    <route id="N_S" edges="n_t t_s"/>\n')
        routes.write('    <route id="S_N" edges="s_t t_n"/>\n')
        routes.write('    <route id="E_W" edges="e_t t_w"/>\n')
        routes.write('    <route id="W_E" edges="w_t t_e"/>\n\n')
        
        # --- BASELINE FLOWS (Constant light traffic for all 4 directions) ---
        routes.write(f'    <flow id="base_NS" route="N_S" begin="0" end="{max_steps}" vehsPerHour="1000" type="car"/>\n')
        routes.write(f'    <flow id="base_SN" route="S_N" begin="0" end="{max_steps}" vehsPerHour="1000" type="car"/>\n')
        routes.write(f'    <flow id="base_EW" route="E_W" begin="0" end="{max_steps}" vehsPerHour="1000" type="car"/>\n')
        routes.write(f'    <flow id="base_WE" route="W_E" begin="0" end="{max_steps}" vehsPerHour="1000" type="car"/>\n\n')
        
        # --- THE RANDOMIZED ADAPTIVE SPIKE ---
        if spike_axis == "EW":
            routes.write(f'    \n')
            routes.write(f'    <flow id="spike_EW" route="E_W" begin="{spike_start}" end="{spike_end}" vehsPerHour="{spike_size}" type="car"/>\n')
            routes.write(f'    <flow id="spike_WE" route="W_E" begin="{spike_start}" end="{spike_end}" vehsPerHour="{spike_size}" type="car"/>\n')
        else:
            routes.write(f'    \n')
            routes.write(f'    <flow id="spike_NS" route="N_S" begin="{spike_start}" end="{spike_end}" vehsPerHour="{spike_size}" type="car"/>\n')
            routes.write(f'    <flow id="spike_SN" route="S_N" begin="{spike_start}" end="{spike_end}" vehsPerHour="{spike_size}" type="car"/>\n')
            
        routes.write('</routes>\n')
        
    print(f"Episode Routes Generated: Spike on {spike_axis} axis from {spike_start}s to {spike_end}s (Intensity: {spike_prob})")