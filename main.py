import os
from engine import process_video
from football_match_analytics import InteractivePlayer

def main():
    video_path = 'high_res_video.f399.mp4' # High resolution video for demonstration
    final_data_path = 'stubs/final_tracks.pkl'
    output_path = 'professional_output.mp4'
    selection_script = 'selections.json'
    
    print("="*60)
    print(" FOOTBALL MATCH INTELLIGENCE SYSTEM - PROFESSIONAL DASHBOARD")
    print("="*60)

    # Step 1: Run the Analytics Engine
    # Note: If you want to force re-computation, delete 'stubs/final_tracks.pkl'
    if not os.path.exists(final_data_path):
        print("\n[STEP 1] Generating Analytics Data (Engine Pass)...")
        process_video(video_path)
    else:
        print("\n[STEP 1] Analytics data found! Skipping engine computation.")
        print(f"         (Delete '{final_data_path}' to recompute)")

    # Step 2: Render the Professional Dashboard
    print(f"\n[STEP 2] Launching Interactive Mode...")
    print("         - Click on a player to select them.")
    print("         - Press 'N' for next player, 'P' for previous.")
    print("         - Press 'ESC' to clear selection.")
    
    player = InteractivePlayer(
        video_path=video_path,
        data_path=final_data_path,
        interactive=True, # Interactive UI mode enabled
        selection_script='', # Disabling the pre-scripted automation
        out_path=output_path
    )
    player.play()
    
    print(f"\n[SUCCESS] Professional video demonstration exported to: {output_path}")

if __name__ == '__main__':
    main()
