import subprocess
import os


class Movie():
    """ Home made class to generate mp4 """

    def __init__(self, fig, name='movie', framerate=30):
        """ input: fig is the handle to the figure """
        self.fig = fig
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        # Open an ffmpeg process
        outf = '%s.mp4' % name
        videoencoder = None
        for v in ['ffmpeg', 'avconv']:
            if subprocess.call(['which', v], stdout=subprocess.PIPE) == 0:
                videoencoder = v

        if videoencoder is None:
            print('\n')
            print('Neither avconv or ffmpeg was found')
            print('Install one or set param.generate_mp4 = False')
            raise ValueError('Install avconv or ffmpeg')

        cmdstring = (videoencoder,
                     '-y', '-r', str(framerate),  # overwrite, 30fps
                     # size of image string
                     '-s', '%dx%d' % (canvas_width, canvas_height),
                     '-pix_fmt', 'argb',  # format
                     '-f', 'rawvideo',
                     # tell ffmpeg to expect raw video from the pipe
                     '-i', '-',
                     '-vcodec', 'libx264', outf)  # output encoding

        devnull = open(os.devnull, 'wb')
        self.process = subprocess.Popen(cmdstring,
                                        stdin=subprocess.PIPE,
                                        stdout=devnull,
                                        stderr=devnull)

    def addframe(self):
        string = self.fig.canvas.tostring_argb()
        self.process.stdin.write(string)

    def finalize(self):
        self.process.communicate()


if __name__ == '__main__':
    """

    Below is an example of how to use this module

    1/ prepare a figure (get the handles of it)
    2/ create the movie instance
    3/ do a loop and update all the handles
    4/ add each frame
    5/ finalize at then end of the loop

    """
    import numpy as np
    import matplotlib.pyplot as plt

    def f(t):
        """ the function to animate """
        return np.sin(2*np.pi*(x-t))

    # Create the plot
    x = np.linspace(-1, 1, 201)
    t = 0.
    template = 'time = %.2f'
    fig = plt.figure()
    p = plt.plot(x, f(t))
    plt.axis([-1, 1, -1, 1])
    p = p[0]
    plt.xlabel('x')
    plt.grid()
    ti = plt.title(template % t)

    # note that you don't even need to plot on the screen
    # to generate the movie!

    # create a Movie instance
    movie = Movie(fig, name='test_moving_sine')

    nt = 100
    dt = 1./nt
    for kt in range(nt):
        t = kt*dt
        # it's faster to update handles
        # than to recreate a plot!
        p.set_data(x, f(t))
        ti.set_text(template % t)
        fig.canvas.draw()
        # add the frame
        movie.addframe()

    # close the movie cleanly (mandatory)
    movie.finalize()
