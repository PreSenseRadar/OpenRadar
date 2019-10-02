import cv2
import numpy as np

win_name = 'track'
win_size = 950
radius = 50
inner_area = win_size - (2 * radius)

min_x = -6
min_y = 0
min_vx = -.5
min_vy = -.5
max_x = 6
max_y = 6
max_vx = .5
max_vy = .5
vec_scaling = 75

BLACK = (0, 0, 0)
DARK_GREY = (50, 50, 50)
GREY = (122, 122, 122)
LIGHT_GREY = (200, 200, 200)
WHITE = (255, 255, 255)

BLUE = (255, 0, 0)

font = cv2.FONT_HERSHEY_SIMPLEX

smiley = cv2.imread('./smiley.png')
smiley_mask = np.zeros(smiley.shape)
smiley_width = smiley.shape[1]
smiley_height = smiley.shape[0]

cv2.circle(smiley_mask, (smiley_width // 2, smiley_height // 2), 200, WHITE, -1)
smiley[smiley_mask != 255] = 0
smiley = smiley[smiley_width // 2 - 200:smiley_width // 2 + 200, smiley_width // 2 - 200:smiley_width // 2 + 200]
smiley = cv2.resize(smiley, (radius * 2, radius * 2))


def norm_y(y):
    """Normalized y from 0-1

    Args:
        y (float): Raw y value

    Returns:
        float: Normalized y value

    """
    return (y - min_y) / (max_y - min_y)


def norm_x(x):
    """Normalized x from 0-1

    Args:
        x (float): Raw x value

    Returns:
        float: Normalized x value

    """
    return (x - min_x) / (max_x - min_x)


def view_y(y):
    """Project y onto image

    Args:
        y (float): Normalized y value

    Returns:
        int: Scaled y image coordinate

    """
    return win_size - (int(inner_area * norm_y(y)) + radius)


def view_x(x):
    """Project x onto image

    Args:
        x (float): Normalized x value

    Returns:
        int: Scaled x image coordinate

    """
    return int(inner_area * norm_x(x)) + radius


# Generate base frame
base_frame = np.ones((win_size, win_size, 3), dtype=np.uint8) * 255
mx = radius + inner_area
# Grid lines
cv2.putText(base_frame, 'y\\x', (5, 20), font, 1, BLACK, 1, cv2.LINE_AA)
for line_y in range(int(max_y + 1)):
    cv2.line(base_frame, (radius, view_y(line_y)), (mx, view_y(line_y)), LIGHT_GREY, 3)
    cv2.putText(base_frame, f'{line_y}[m]', (5, view_y(line_y) + 5), font, .5, BLACK, 1, cv2.LINE_AA)

for line_x in range(int(min_x), int(max_x + 1)):
    cv2.line(base_frame, (view_x(line_x), radius), (view_x(line_x), mx), LIGHT_GREY, 3)
    cv2.putText(base_frame, f'{line_x}[m]', (view_x(line_x) - 20, radius - 10), font, .5, BLACK, 1, cv2.LINE_AA)

# Borders
cv2.line(base_frame, (radius, radius), (radius, mx), BLACK, 7)
cv2.line(base_frame, (radius, mx), (mx, mx), BLACK, 7)
cv2.line(base_frame, (mx, mx), (mx, radius), BLACK, 7)
cv2.line(base_frame, (mx, radius), (radius, radius), BLACK, 7)

# Radar azimuth view
cv2.line(base_frame, (view_x(0), view_y(0)), (view_x(max_x), view_y(3.46)), BLACK, 3)
cv2.line(base_frame, (view_x(0), view_y(0)), (view_x(min_x), view_y(3.46)), BLACK, 3)

c_fill = np.copy(base_frame)
cv2.circle(c_fill, (view_x(0), view_y(0)), view_y(min_y) - view_y(max_y) - 100, BLACK, 3)
base_frame[:, view_x(min_x):view_x(max_x)] = c_fill[:, view_x(min_x):view_x(max_x)]


def get_empty_frame():
    """Generate a copy of the base frame

    Returns:
        ndarray: Base annotated frame to modify

    """
    global base_frame
    return np.copy(base_frame)


def draw_points(points, m_num, frame=None):
    """Draw raw data points onto the frame

    Args:
        points (list): List of point objects detected
        m_num (int): Number of points detected
        frame (ndarray): Frame to modify

    Returns:
        ndarray: Modified frame

    """
    for point, _ in zip(points, range(m_num)):
        r = point.range
        a = point.angle
        x_pos = -np.sin(a) * r
        y_pos = np.cos(a) * r
        try:
            cv2.circle(frame, (view_x(x_pos), view_y(y_pos)), 3, (0, 0, 255), -1)
        except:
            pass
    return frame


def draw_objs(points, frame=None, c_color=(0, 0, 255), l_color=(255, 255, 255)):
    """

    Args:
        points (list): List of the information about each object
        frame (ndarray): Frame to modify
        c_color (tuple): Color of the circles representing groupings
        l_color (tuple): Color of the velocity vector representation

    Returns:
        ndarray: Modified frame

    """
    x_pos, y_pos, x_vel, y_vel = points
    x_norm = (x_pos - min_x) / (max_x - min_x)
    y_norm = (y_pos - min_y) / (max_y - min_y)
    vx_norm = ((x_vel - min_vx) / (max_vx - min_vx)) - .5
    vy_norm = ((y_vel - min_vy) / (max_vy - min_vy)) - .5
    x_scaled = int(inner_area * x_norm) + radius
    y_scaled = int(inner_area * y_norm) + radius
    cv2.circle(frame, (x_scaled, y_scaled), radius, c_color, -1)

    x_vec = int(inner_area * (vec_scaling * vx_norm))
    y_vec = int(inner_area * (vec_scaling * vy_norm))
    cv2.arrowedLine(frame, (x_scaled, y_scaled), (x_scaled + x_vec, y_scaled + y_vec), l_color, 10)

    return frame


def update_frame(target_desc, num_trackers, frame=None):
    """Draw the detected objects onto the frame

    Args:
        target_desc (list): List of target objects
        num_trackers (int): Number of detected objects
        frame (ndarray): Frame to modify

    Returns:
        ndarray: Modified frame

    """
    global min_x, min_y, min_vx, min_vy
    global max_x, max_y, max_vx, max_vy
    if frame is None:
        frame = np.zeros((win_size, win_size, 3), dtype=np.uint8)

    for t, tid in zip(target_desc, range(num_trackers)):
        x_pos, y_pos, x_vel, y_vel = t.S[:4]
        x_pos = -x_pos
        try:
            draw_img(frame, view_x(x_pos), view_y(y_pos), smiley)
            cv2.putText(frame, f'ID:{tid}', (view_x(x_pos) - 30, view_y(y_pos) - 60), font, 1, BLACK, 2, cv2.LINE_AA)

            vec_mag = np.sqrt(x_vel ** 2 + y_vel ** 2)
            if vec_mag < .3:
                continue
            x_unit = int((x_vel / vec_mag) * vec_scaling)
            y_unit = int((y_vel / vec_mag) * vec_scaling)
            cv2.arrowedLine(frame, (view_x(x_pos), view_y(y_pos)),
                            (view_x(x_pos) - x_unit, view_y(y_pos) - y_unit),
                            BLUE, 3)
        except Exception as e:
            print(e)
            pass

    return frame


def draw_img(frame, x, y, img):
    """Draw an image onto a frame

    Args:
        frame: Frame to modify
        x: Centered x coordinate on frame
        y: Centered y coordinate on frame
        img: Image to draw

    Returns:
        ndarray: Modified frame

    """
    hw = img.shape[1] // 2
    hh = img.shape[0] // 2
    snip = frame[y - hh:y + hh, x - hw:x + hw]
    snip[img != 0] = img[img != 0]


def show(frame, wait=100):
    """Display the frame

    Args:
        frame (ndarray): Frame to display
        wait (int): Wait time until moving on

    Returns:
        bool: False if user requested a break, true otherwise

    """
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(wait)
    if key == ord('q'):
        return False
    return True


def destroy():
    """Destroy all open-cv generated windows

    Returns:
        None

    """
    cv2.destroyAllWindows()
