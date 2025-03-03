import matplotlib.pyplot as plt


def generate_patch_size_plot():
    """
    Generate a plot of inference time (in seconds) vs. patch size.
    """
    # Define the x and y values
    patch_sizes = [2, 4, 8, 16]
    times = [906, 238, 145, 127]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(patch_sizes, times, marker='o', linestyle='--', color='blue', label='Inference Time')

    # Label the axes and add a title
    ax.set_xlabel("Patch Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Inference Time vs. Patch Size")

    # Add grid and legend for clarity
    ax.grid(True)
    ax.legend()

    # Save the figure
    fig.savefig("patch_size_time_p4_q4.png")

    # Display the plot
    plt.show()


if __name__ == '__main__':
    generate_patch_size_plot()
