def shave_ends(signal, annotations):
    """
    This function shaves off the non-sinus beats at the beginning and at the end, so that we are not looking at
    negative indices or beyond the length of the signal. Any signal can be shaved, including the annotations themselves,
    which may be very useful
    :param signal: an iterable, sliceable object
    :param annotations: an interable, sliceable object containing the annotations, 0 is considered to be of sinus
    origin
    :return: shaved signal
    """
    try:
        # removing nonsinus beats from the beginning
        while annotations[0] != 0:
            signal = signal[1:]
            annotations = annotations[1:]
        # removing nonsinus beats from the end
        while annotations[-1] != 0:
            signal = signal[0:-1]
            annotations = annotations[0:-1]
    except IndexError:
        print("no good beats")
    return signal
