import math


__author__ = "Daniel Lindsley"
__license__ = "New BSD"
__version__ = (1, 1, 0)


def euclidean_compare(ref_point, check_point):
    """
    Calculates a raw euclidean value for comparison with other raw values.
    This calculates the sum of the delta of X values plus the delta of Y
    values. It skips the square root portion of the Pythagorean theorem,
    for speed.
    If you need a proper euclidean distance value, see `euclidean_distance`.
    Primarily for internal use, but stable API if you need it.
    Args:
        ref_point (Point): The first point to check.
        check_point (Point): The second point to check.
    Returns:
        int|float: The sum value.
    """
    dx = max(ref_point.x, check_point.x) - min(ref_point.x, check_point.x)
    dy = max(ref_point.y, check_point.y) - min(ref_point.y, check_point.y)
    return dx ** 2 + dy ** 2


def euclidean_distance(ref_point, check_point):
    """
    Calculates a euclidean distance between points.
    Args:
        ref_point (Point): The first point to check.
        check_point (Point): The second point to check.
    Returns:
        int|float: The (unitless) distance value.
    """
    return math.sqrt(euclidean_compare(ref_point, check_point))


def visualize(tree, size=10):  # pragma: no cover
    """
    Using `matplotlib`, generates a visualization of the `QuadTree`.
    You will have to separately install `matplotlib`, as this library does
    not depend on it in any other way::
        $ pip install matplotlib
    Once installed, this will automatically generate an entire plot of all
    the points within, as well as lines for the subdivisions of nodes.
    Args:
        tree (`QuadTree`): The quadtree itself.
        size (int): The size of the resulting output diagram.
    """
    from matplotlib import pyplot

    def draw_all_nodes(node):
        for pnt in node.points:
            pyplot.plot(pnt.x, pnt.y, ".")

        if node.ul:
            draw_lines(node)
            draw_all_nodes(node.ul)
        if node.ur:
            draw_all_nodes(node.ur)
        if node.ll:
            draw_all_nodes(node.ll)
        if node.lr:
            draw_all_nodes(node.lr)

    def draw_lines(node):
        bb = node.bounding_box

        # The scales for axhline & axvline are 0-1, so we have to convert
        # our values.
        x_offset = -tree._root.bounding_box.min_x
        min_x = (bb.min_x + x_offset) / 100
        max_x = (bb.max_x + x_offset) / 100

        y_offset = -tree._root.bounding_box.min_y
        min_y = (bb.min_y + y_offset) / 100
        max_y = (bb.max_y + y_offset) / 100

        pyplot.axhline(
            node.center.y, min_x, max_x, color="grey", linewidth=0.5
        )
        pyplot.axvline(
            node.center.x, min_y, max_y, color="grey", linewidth=0.5
        )

    pyplot.figure(figsize=(size, size))

    # Draw the axis first.
    half_width = tree.width / 2
    half_height = tree.height / 2
    min_x, max_x = tree.center.x - half_width, tree.center.x + half_width
    min_y, max_y = (
        tree.center.y - half_height,
        tree.center.y + half_height,
    )
    pyplot.axis([min_x, max_x, min_y, max_y])

    draw_all_nodes(tree._root)
    pyplot.show()


class Point(object):
    """
    An object representing X/Y cartesean coordinates.
    """

    def __init__(self, x, y, data=None):
        """
        Constructs a `Point` object.
        Args:
            x (int|float): The X coordinate.
            y (int|float): The Y coordinate.
            data (any): Optional. Corresponding data for that point. Default
                is `None`.
        """
        self.x = x
        self.y = y
        self.data = data

    def __repr__(self):
        return "<Point: ({}, {})>".format(self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        """
        Checks if a point's *coordinates* are equal to another point's.
        This does **NOT** ensure the data is the same. This library doesn't
        concern itself with what data you're storing on the points.
        Args:
            other (Point): The other point to check against.
        Returns:
            bool: `True` if the coordinates match, otherwise `False`.
        """
        return self.x == other.x and self.y == other.y


class BoundingBox(object):
    """
    A object representing a bounding box.
    """

    def __init__(self, min_x, min_y, max_x, max_y):
        """
        Constructs a `Point` object.
        Args:
            min_x (int|float): The minimum X coordinate.
            min_y (int|float): The minimum Y coordinate.
            max_x (int|float): The maximum X coordinate.
            max_y (int|float): The maximum Y coordinate.
        """
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.half_width = self.width / 2
        self.half_height = self.height / 2
        self.center = Point(self.half_width, self.half_height)

    def __repr__(self):
        return "<BoundingBox: ({}, {}) to ({}, {})>".format(
            self.min_x, self.min_y, self.max_x, self.max_y
        )

    def contains(self, point):
        """
        Checks if a point is within the bounding box.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if the point is within the box, otherwise `False`.
        """
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
        )

    def intersects(self, other_bb):
        """
        Checks if another bounding box intersects with this bounding box.
        Args:
            other_bb (BoundingBox): The bounding box to check.
        Returns:
            bool: `True` if they intersect, otherwise `False`.
        """
        return not (
            other_bb.min_x > self.max_x
            or other_bb.max_x < self.min_x
            or other_bb.max_y < self.min_y
            or other_bb.min_y > self.max_y
        )


class QuadNode(object):
    """
    A node within the QuadTree.
    Typically, you won't use this object directly. The `QuadTree` object
    provides a more convenient API. However, if you know what you're doing
    or need to customize, `QuadNode` is here.
    """

    POINT_CAPACITY = 4
    point_class = Point
    bb_class = BoundingBox

    def __init__(self, center, width, height, capacity=None):
        """
        Constructs a `QuadNode` object.
        Args:
            center (tuple|Point): The center point of the quadtree.
            width (int|float): The width of the point space.
            height (int|float): The height of the point space.
            capacity (int): Optional. The number of points per quad before
                subdivision occurs. Default is `None`, which defers to
                `QuadNode.POINT_CAPACITY`, which is `4`.
        """
        self.center = center
        self.width = width
        self.height = height
        self.points = []

        self.ul = None
        self.ur = None
        self.ll = None
        self.lr = None

        if capacity is None:
            capacity = self.POINT_CAPACITY

        self.capacity = capacity
        self.bounding_box = self._calc_bounding_box()

    def __repr__(self):
        return "<QuadNode: ({}, {}) {}x{}>".format(
            self.center.x, self.center.y, self.width, self.height
        )

    def __contains__(self, point):
        """
        Checks if a point is found within the node's data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it found, otherwise `False`.
        """
        return self.find(point) is not None

    def __len__(self):
        """
        Returns a count of how many points are in the node.
        Returns:
            int: A count of all the points.
        """
        count = len(self.points)

        if self.ul is not None:
            count += len(self.ul)

        if self.ur is not None:
            count += len(self.ur)

        if self.ll is not None:
            count += len(self.ll)

        if self.lr is not None:
            count += len(self.lr)

        return count

    def __iter__(self):
        """
        Iterates (lazily) over all the points located within a node &
        its children.
        Returns:
            iterable: All the `Point` objects.
        """
        # Make sure we slice it, so that we copy the whole list & don't
        # risk modifying the original.
        for pnt in self.points[:]:
            yield pnt

        if self.ul is not None:
            yield from self.ul

        if self.ur is not None:
            yield from self.ur

        if self.ll is not None:
            yield from self.ll

        if self.lr is not None:
            yield from self.lr

    def _calc_bounding_box(self):
        half_width = self.width / 2
        half_height = self.height / 2

        min_x = self.center.x - half_width
        min_y = self.center.y - half_height
        max_x = self.center.x + half_width
        max_y = self.center.y + half_height

        return self.bb_class(
            min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y
        )

    def contains_point(self, point):
        """
        Checks if a point would be within the bounding box of the node.
        This is a bounding check, not verification the point is present in
        the data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it is within the bounds, otherwise `False`.
        """
        bb = self.bounding_box

        if bb.min_x <= point.x <= bb.max_x:
            if bb.min_y <= point.y <= bb.max_y:
                return True

        return False

    def is_ul(self, point):
        """
        Checks if a point would be in the upper-left quadrant of the node.
        This is a bounding check, not verification the point is present in
        the data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it would be, otherwise `False`.
        """
        return point.x < self.center.x and point.y >= self.center.y

    def is_ur(self, point):
        """
        Checks if a point would be in the upper-right quadrant of the node.
        This is a bounding check, not verification the point is present in
        the data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it would be, otherwise `False`.
        """
        return point.x >= self.center.x and point.y >= self.center.y

    def is_ll(self, point):
        """
        Checks if a point would be in the lower-left quadrant of the node.
        This is a bounding check, not verification the point is present in
        the data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it would be, otherwise `False`.
        """
        return point.x < self.center.x and point.y < self.center.y

    def is_lr(self, point):
        """
        Checks if a point would be in the lower-right quadrant of the node.
        This is a bounding check, not verification the point is present in
        the data.
        Args:
            point (Point): The point to check.
        Returns:
            bool: `True` if it would be, otherwise `False`.
        """
        return point.x >= self.center.x and point.y < self.center.y

    def subdivide(self):
        """
        Subdivides an existing node into the node + children.
        Returns:
            None: Nothing to see here. Please go about your business.
        """
        half_width = self.width / 2
        half_height = self.height / 2
        quarter_width = half_width / 2
        quarter_height = half_height / 2

        ul_center = self.point_class(
            self.center.x - quarter_width, self.center.y + quarter_height
        )
        self.ul = self.__class__(
            ul_center, half_width, half_height, capacity=self.capacity
        )

        ur_center = self.point_class(
            self.center.x + quarter_width, self.center.y + quarter_height
        )
        self.ur = self.__class__(
            ur_center, half_width, half_height, capacity=self.capacity
        )

        ll_center = self.point_class(
            self.center.x - quarter_width, self.center.y - quarter_height
        )
        self.ll = self.__class__(
            ll_center, half_width, half_height, capacity=self.capacity
        )

        lr_center = self.point_class(
            self.center.x + quarter_width, self.center.y - quarter_height
        )
        self.lr = self.__class__(
            lr_center, half_width, half_height, capacity=self.capacity
        )

        # Redistribute the points.
        # Manually call `append` here, as calling `.insert()` creates an
        # infinite recursion situation.
        for pnt in self.points:
            if self.is_ul(pnt):
                self.ul.points.append(pnt)
            elif self.is_ur(pnt):
                self.ur.points.append(pnt)
            elif self.is_ll(pnt):
                self.ll.points.append(pnt)
            else:
                self.lr.points.append(pnt)

        self.points = []

    def insert(self, point):
        """
        Inserts a `Point` into the node.
        If the node exceeds the maximum capacity, it will subdivide itself
        & redistribute its points before adding the new one. This means there
        can be some variance in the performance of this method.
        Args:
            point (Point): The point to insert.
        Returns:
            bool: `True` if insertion succeeded, otherwise `False`.
        """
        if not self.contains_point(point):
            raise ValueError(
                "Point {} is not within this node ({} - {}).".format(
                    point, self.center, self.bounding_box
                )
            )

        # Check to ensure we're not going to go over capacity.
        if (len(self.points) + 1) > self.capacity:
            # We're over capacity. Subdivide, then insert into the new child.
            self.subdivide()

        if self.ul is not None:
            if self.is_ul(point):
                return self.ul.insert(point)
            elif self.is_ur(point):
                return self.ur.insert(point)
            elif self.is_ll(point):
                return self.ll.insert(point)
            elif self.is_lr(point):
                return self.lr.insert(point)

        # There are no child nodes & we're under capacity. Add it to `points`.
        self.points.append(point)
        return True

    def find(self, point):
        """
        Searches for the node that would contain the `Point` within the
        node & it's children.
        Args:
            point (Point): The point to search for.
        Returns:
            Point|None: Returns the `Point` (including it's data) if found.
                `None` if the point is not found.
        """
        found_node, _ = self.find_node(point)

        if found_node is None:
            return None

        # Try the points on this node first.
        for pnt in found_node.points:
            if pnt.x == point.x and pnt.y == point.y:
                return pnt

        return None

    def remove(self, point):
        found_node, _ = self.find_node(point)
        if found_node is None:
            return None

        for pnt in found_node.points:
            if pnt.x == point.x and pnt.y == point.y:
                found_node.points.remove(pnt)

        return None

    def find_node(self, point, searched=None):
        """
        Searches for the node that would contain the `Point` within the
        node & it's children.
        Args:
            point (Point): The point to search for.
            searched (list|None): Optional. This is a list of all the nodes
                that were touched during the search. Default is `None`, which
                will construct an empty `list` to pass to recursive calls.
        Returns:
            tuple: (QuadNode|None, list): Returns the node where the point
                would be found or `None`, AND the list of nodes touched
                during the search.
        """
        if searched is None:
            searched = []

        if not self.contains_point(point):
            return None, searched

        searched.append(self)

        # Check the children.
        if self.is_ul(point):
            if self.ul is not None:
                return self.ul.find_node(point, searched)
        elif self.is_ur(point):
            if self.ur is not None:
                return self.ur.find_node(point, searched)
        elif self.is_ll(point):
            if self.ll is not None:
                return self.ll.find_node(point, searched)
        elif self.is_lr(point):
            if self.lr is not None:
                return self.lr.find_node(point, searched)

        # Not found in any children. Return this node.
        return self, searched

    def all_points(self):
        """
        Returns a **list** of all the points located within a node &
        its children.
        Returns:
            list: All the `Point` objects in an unordered list.
        """
        return list(iter(self))

    def within_bb(self, bb):
        """
        Checks if a bounding box is within the node's bounding box.
        Primarily for internal use, but stable API if you need it.
        Args:
            bb (BoundingBox): The bounding box to check.
        Returns:
            bool: `True` if the bounding boxes intersect, otherwise `False`.
        """
        points = []

        # If we don't intersect with the bounding box, return an empty list.
        if not self.bounding_box.intersects(bb):
            return points

        # Check if any of the points on this instance are within the BB.
        for pnt in self.points:
            if bb.contains(pnt):
                points.append(pnt)

        if self.ul is not None:
            points += self.ul.within_bb(bb)

        if self.ur is not None:
            points += self.ur.within_bb(bb)

        if self.ll is not None:
            points += self.ll.within_bb(bb)

        if self.lr is not None:
            points += self.lr.within_bb(bb)

        return points


class QuadTree(object):

    node_class = QuadNode
    point_class = Point

    def __init__(self, center, width, height, capacity=None):
        """
        Constructs a `QuadTree` object.
        Args:
            center (tuple|Point): The center point of the quadtree.
            width (int|float): The width of the point space.
            height (int|float): The height of the point space.
            capacity (int): Optional. The number of points per quad before
                subdivision occurs. Default is `None`.
        """
        self.width = width
        self.height = height
        self.center = self.convert_to_point(center)
        self._root = self.node_class(
            self.center, self.width, self.height, capacity=capacity
        )

    def __repr__(self):
        return "<QuadTree: ({}, {}) {}x{}>".format(
            self.center.x, self.center.y, self.width, self.height,
        )

    def convert_to_point(self, val):
        """
        Converts a value to a `Point` object.
        This is to allow shortcuts, like providing a tuple for a point.
        Args:
            val (Point|tuple|None): The value to convert.
        Returns:
            Point: A point object.
        """
        if isinstance(val, self.point_class):
            return val
        elif isinstance(val, (tuple, list)):
            return self.point_class(val[0], val[1])
        elif val is None:
            return self.point_class(0, 0)
        else:
            raise ValueError(
                "Unknown data provided for point. Please use one of: "
                "quads.Point | tuple | list | None"
            )

    def __contains__(self, point):
        """
        Checks if a `Point` is found in the quadtree.
        > Note: This doesn't check if a point is within the bounds of the
        > tree, but if that *specific point* is in the tree.
        Args:
            point (Point|tuple|None): The point to check for.
        Returns:
            bool: `True` if found, otherwise `False`.
        """
        pnt = self.convert_to_point(point)
        return self.find(pnt) is not None

    def __len__(self):
        """
        Returns a count of how many points are in the tree.
        Returns:
            int: A count of all the points.
        """
        return len(self._root)

    def __iter__(self):
        """
        Returns an iterator for all the points in the tree.
        Returns:
            iterator: An iterator of all the points.
        """
        return iter(self._root)

    def insert(self, point, data=None):
        """
        Inserts a `Point` into the quadtree.
        Args:
            point (Point|tuple|None): The point to insert.
            data (any): Optional. Corresponding data for that point. Default
                is `None`.
        Returns:
            bool: `True` if insertion succeeded, otherwise `False`.
        """
        pnt = self.convert_to_point(point)
        pnt.data = data
        return self._root.insert(pnt)

    def find(self, point):
        """
        Searches for a `Point` within the quadtree.
        Args:
            point (Point|tuple|None): The point to search for.
        Returns:
            Point|None: Returns the `Point` (including it's data) if found.
                `None` if the point is not found.
        """
        pnt = self.convert_to_point(point)
        return self._root.find(pnt)

    def remove(self, point):
        pnt = self.convert_to_point(point)
        return self._root.remove(pnt)

    def within_bb(self, bb):
        """
        Checks if a bounding box is within the quadtree's bounding box.
        Primarily for internal use, but stable API if you need it.
        Args:
            bb (BoundingBox): The bounding box to check.
        Returns:
            bool: `True` if the bounding boxes intersect, otherwise `False`.
        """
        return self._root.within_bb(bb)

    def nearest_neighbors(self, point, count=10):
        """
        Returns the nearest points of a given point, sorted by distance
        (closest first).
        The desired point does not need to exist within the quadtree, but
        does need to be within the tree's boundaries.
        Args:
            point (Point): The desired location to search around.
            count (int): Optional. The number of neighbors to return. Default
                is `10`.
        Returns:
            list: The nearest `Point` neighbors.
        """
        # Algorithm description:
        # * Search down to find the smallest node around the desired point,
        #   retaining a stack of nodes visited on the way down.
        # * Reverse the visited stack, so that it's now in
        #   smallest/closest-to-largest/furthest order.
        # * Iterate over the node stack.
        #   * Collect the points from the current node & it's children.
        #   * Sort the points by euclidean distance, using
        #     `euclidean_compare`, since the actual distance doesn't matter
        #     for now.
        #   * Add them to the "found" results.
        #   * If the "found" count is greater-than-or-equal to the desired
        #     count, break out of the loop.
        # * If the stack is exhausted, we have all the points in the entire
        #   quadtree & can just return them.
        # * Otherwise, we now have a decent set of results, ordered by
        #   distance. But we are not done. It's possible/probable that there
        #   are other nearby quadnodes that weren't touched by the search
        #   BUT are physically closer.
        # * Take our furthest point and use it as a radius for a search
        #   "circle".
        #     * We'll actually just create a bounding box, which is
        #       computationally cheaper & we already have methods that
        #       support it.
        #     * Using that radius as a distance to the *edge* (not a corner),
        #       we create a box big enough to fit the search circle.
        # * Collect all the points within that bounding box.
        # * Re-sort them by euclidean distance (again, using
        #   `euclidean_compare`).
        # * Slice it to match the desired count & return them.

        point = self.convert_to_point(point)
        nearest_results = []

        # Check to see if it's within our bounds first.
        if not self._root.contains_point(point):
            return nearest_results

        # First, find the target node.
        node, searched_nodes = self._root.find_node(point)

        # Reverse the order, as they come back in coarse-to-fine order, which
        # is the opposite of nearby points.
        searched_nodes.reverse()
        seen_nodes = set()
        seen_points = set()

        # From here, we'll work our way backwards out through the nodes.
        for node in searched_nodes:
            # Mark the node as already checked.
            seen_nodes.add(node)
            local_points = []

            for pnt in node.all_points():
                if pnt in seen_points:
                    continue

                seen_points.add(pnt)
                local_points.append(pnt)

            local_points = sorted(
                local_points, key=lambda lpnt: euclidean_compare(point, lpnt)
            )
            nearest_results.extend(local_points)

            if len(nearest_results) >= count:
                break

        # Slice off any extras.
        nearest_results = nearest_results[:count]

        if len(seen_nodes) == len(searched_nodes):
            # We've exhausted everything. Return what we've got.
            return nearest_results[:count]

        search_radius = euclidean_distance(point, nearest_results[-1])
        search_bb = BoundingBox(
            point.x - search_radius,
            point.y - search_radius,
            point.x + search_radius,
            point.y + search_radius,
        )
        bb_results = self._root.within_bb(search_bb)
        nearest_results = sorted(bb_results, key=lambda lpnt: euclidean_compare(point, lpnt))

        return nearest_results[:count]