source_points = np.float32([[361., 212.],
                            [673., 248.],
                            [597., 338.],
                            [265., 286.]])

for point in source_points:
    cv2.circle(original_image_RGB_copy, tuple(point), 8, (255, 0, 0), -1)

points = source_points.reshape((-1, 1, 2)).astype(np.int32)
cv2.polylines(original_image_RGB_copy, [
              points], True, (0, 255, 0), thickness=4)


src = source_points
dst = np.float32([(0.49, 0.5), (0.77, 0.5), (0.77, 0.65), (0.49, 0.65)])
dst_size = (800, 1080)
dst = dst * np.float32(dst_size)

H_matrix = cv2.getPerspectiveTransform(src, dst)
print("The perspective transform matrix:")
print(H_matrix)

warped = cv2.warpPerspective(original_image_RGB_copy, H_matrix, dst_size)


def compute_point_perspective_transformation(matrix, boxes):
    list_downoids = [[box[4], box[5]+box[3]//2] for box in boxes]
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(
        list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append(
            [transformed_points[i][0][0], transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')


def get_birds_eye_view_image(green_box, red_box, eye_view_height, eye_view_width):
    blank_image = cv2.imread('templates/black_background.png')

    cv2.putText(blank_image, str(len(red_box)), (120, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(blank_image, str(len(green_box)), (520, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    for point in green_box:
        cv2.circle(blank_image, tuple(
            [point[6], point[7]]), 20, (0, 255, 0), -1)
    for point in red_box:
        cv2.circle(blank_image, tuple(
            [point[6], point[7]]), 20, (0, 0, 255), -1)
    blank_image = cv2.resize(blank_image, (eye_view_width, eye_view_height))
    return blank_image
