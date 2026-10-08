# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))

    sqrt3_half = np.sqrt(3) / 2

    # Dense hexagonal grid: 4 rows of 7,6,7,6 = 26 circles
    # Optimized spacing: balance horizontal/vertical so max equal radius is
    # limited by the grid, not by uneven axis scaling.
    # Target: equal-circle packing. If dx = 1/7 and dy = dx * sqrt3/2,
    # total vertical span = 3*dy ~ 0.371. Instead, scale so both axes are tight:
    # 7 circles fit in width => dx = 1/7 => r = 1/14;
    # vertical: 4 rows give 3 gaps of r*sqrt(3). Total = 3*sqrt(3)/14 ~ 0.371.
    # Vertical slack = (1 - 3*sqrt(3)/14)/2 ~ 0.3144 -- we could grow circles.
    # To maximize equal radius, set r = min(1/14, 1/(2 + 2*3*sqrt(3)/2)...) 
    # For equal radius r in 4 rows: 2r + 3*r*sqrt(3) = 1 -> r = 1/(2 + 3*sqrt(3)) ~ 0.1319
    # But horizontal with 7 circles: 14r = 1 -> r = 1/14 ~ 0.0714. So horizontal is tight.
    # Try: use r = 1/14 for horizontal (dx=2r), but allow larger vertical spacing
    # equal to 2r to pack more uniformly -> then total height = 2r + 3*2r = 8r = 8/14 ~ 0.571
    # still vertical slack, but we already reached horizontal limit.
    # Alternatively: 5 rows; try 5 rows of 6,5,6,5,4 = 26? 6+5+6+5+4=26 -> explore mixed.
    # Simpler: optimize the 4-row 7,6,7,6 by pushing the vertical expansion to allow
    # larger radii. That means: set dy = 2*r where r is the target equal radius,
    # but r is bounded by horizontal: 7*2r = 1 -> r = 1/14.
    # So use r = 1/14, dx = 2r = 1/7, dy = sqrt(3)*r = sqrt(3)/14 ~ 0.1237?
    # Wait: hexagonal lattice spacing: adjacent row vertical distance = sqrt(3)*r.
    # If we set r equal to 1/14, then adjacent centers distance = 2r = 1/7.
    # Vertical gap between rows = sqrt(3)/2 * dx = sqrt(3)/14 ~ 0.1237, total height = 3*sqrt(3)/14 ~ 0.371.
    # That leaves huge vertical slack; we could make circles bigger by expanding in y,
    # but horizontal limits r to 1/14.
    #
    # Better: use 5 rows. 5 rows of counts 5,6,5,6,4 = 26.
    # Horizontal: 6 circles span -> 12r <= 1 -> r <= 1/12 ~ 0.0833.
    # Vertical: 4 gaps of sqrt(3)*r + 2r = r*(2 + 4*sqrt(3)) ~ r*8.928 -> r <= 1/8.928 ~ 0.112. 
    # Horizontal tight at r = 1/12 ~ 0.0833, sum = 26/12 ~ 2.167 -- worse than 26/14 ~ 1.857?
    # Wait 26/14 ~ 1.857, 26/12 ~ 2.167. So 5 rows with 6-wide would give bigger sum
    # per equal radius but less total because sum scales linearly. Actually same formula n*r.
    # So maximizing max equal radius r subject to constraints is optimal for equal packing.
    # But packing with varied radii can yield larger sum -- large circles near edges? No,
    # borders limit radius to distance-to-edge.
    #
    # Strategy: Use 4-row 7,6,7,6 grid but allow non-equal radii. Interior circles are
    # limited by neighbors to radius = dx/2 = 1/14. Edge circles have larger potential
    # radius because they can reach the border (min(x, y, 1-x, 1-y) is larger for some).
    # However, edge circles are next to interior circles which cap at 1/14, so edge circles
    # can grow up to dist - 1/14. For corner circles, dist to interior neighbor = dx in
    # x or dy in y; if interior neighbor has radius 1/14, then edge circle radius =
    # dist - 1/14 = dx - 1/14 = 1/7 - 1/14 = 1/14. Same! Actually edge circle center
    # is at distance dx from its interior neighbor; both have radius <= dx/2 = 1/14.
    # So edge circles limited similarly.
    # But corner circles (in grid) have two interior neighbors (one horizontal, one diagonal?),
    # and border distance. In hexagonal 7-circle row, corners at x=dx/2, x=1-dx/2.
    # Border distance = dx/2 = 1/14. So again r = 1/14.
    #
    # To improve: use a larger grid but shift some circles further out. Let's try a denser
    # pattern with 6 rows: 5,5,5,5,3,3 = 26? 5+5+5+5+3+3=26. Actually try 5 rows: 6,5,6,5,4=26.
    # Let r_max for equal = 1/(2*6) = 1/12 horizontally; vertically = 1/(2+4*sqrt(3)) ~ 0.112.
    # So r = 1/12, sum = 26/12 ~ 2.1667.
    # Better: 4 rows give sum 26/14 ~ 1.857; 5 rows 26/12 ~ 2.167; 6 rows? try 5,4,5,4,4,4=26.
    # Horizontal 5 circles -> r = 1/10 = 0.1, vertical 5 gaps r*sqrt(3) => 2r+5sqrt(3)r = r*(2+8.66)=r*10.66, r~0.094.
    # r min(0.1,0.094)=0.094, sum=26*0.094~2.444. Better!
    # Let's use 6 rows 5,4,5,4,4,4=26; or 6 rows 5,4,5,4,5,3=26.

    # We'll use 6 rows: 5,4,5,4,5,3 = 26.
    row_counts = [5, 4, 5, 4, 5, 3]
    n_rows = len(row_counts)

    # Equal radius target: limited horizontally by 5-circle rows. 5 circles in width:
    # span = 10r (from r to 1-r with spacing 2r between centers). So r = 1/10 = 0.1.
    # Vertical: n_rows rows, (n_rows-1) gaps of r*sqrt(3), plus 2r at edges.
    # Total = 2r + (n_rows-1)*sqrt(3)*r = r*(2 + (n_rows-1)*sqrt(3)).
    # For n_rows=6: r*(2 + 5*sqrt(3)) = r*10.660..., so r <= 1/10.660 ~ 0.0938.
    # Use r = 0.0938 and scale both axes to fit tightly.

    r_eq = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3))
    dx = 2.0 * r_eq
    dy = dx * sqrt3_half

    # Total width for 5-circle row: 4 intervals of dx + 2*r_eq = 4*2r + 2r = 10r.
    # Set r_eq so 10r <= 1 -> r <= 0.1. We already have r = 0.0938 from vertical,
    # and 10r = 0.938 < 1 -- so we have horizontal slack. Tighten horizontally:
    # expand dx until 10r_eq = 1 -> r = 0.1, but then vertical fails.
    # To balance, set r = min(0.1, r_eq) = r_eq ~ 0.0938, and center horizontally.
    # Actually use this r and set horizontal spacing dx=2r; the 5-circle row spans
    # from r to (1-r) naturally with 4*dx = 8r, plus 2r = 10r = 0.938 -> centered.
    # For rows with 4 circles (odd rows offset by dx/2): span = 3*dx = 6r, plus offset
    # and margins: from r + dx/2 to r + dx/2 + 3*dx = r + 3.5dx = r + 7r = 8r = 0.75.
    # Less than 1-r, so they fit centered.
    # For row with 3 circles: span = 2*dx = 4r, plus offsets, total = r + dx/2 + 2*dx = r + 2.5dx = 6r = 0.563.
    # All fit.

    x_start_even = r_eq
    x_start_odd = r_eq + dx / 2.0
    total_height = 2.0 * r_eq + (n_rows - 1) * dy
    y_start = (1.0 - total_height) / 2.0 + r_eq

    idx = 0
    for row_i, count in enumerate(row_counts):
        y = y_start + row_i * dy
        if row_i % 2 == 0:
            x_start = x_start_even
        else:
            x_start = x_start_odd
        for k in range(count):
            centers[idx] = [x_start + k * dx, y]
            idx += 1

    radii = compute_max_radii(centers)

    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Uses iterative pairwise constraint equal-share tightening.
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # Distance to borders first
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively enforce r_i + r_j <= d_ij: equal-share to maximize joint contribution
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
                if radii[i] + radii[j] > dist:
                    new_r = dist / 2.0
                    if radii[i] > new_r:
                        radii[i] = new_r
                        changed = True
                    if radii[j] > new_r:
                        radii[j] = new_r
                        changed = True
        if not changed:
            break

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)