# Import the modules
import pycuda_detection_new
import warp_optimaize_final
import memory_lock
import pycuda_shared

def main():
    image_path = "test.png"

    # Call visualize_all_kernels() from each module
    print("Calling functions from different modules:")
    result_naive = pycuda_detection_new.edge_detection_pipeline(image_path)
    result_warp = warp_optimaize_final.edge_detection_pipeline(image_path)
    result_memlock = memory_lock.edge_detection_pipeline(image_path)
    result_shared = pycuda_shared.edge_detection_pipeline(image_path)
    print("Edge-detected image shape:", result_naive.shape)
    print("Edge-detected image shape:", result_warp.shape)
    print("Edge-detected image shape:", result_memlock.shape)
    print("Edge-detected image shape:", result_shared.shape)

if __name__ == "__main__":
    main()
