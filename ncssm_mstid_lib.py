import numpy as np
import scipy as sp
import datetime

from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib import dates as md
import matplotlib

from mpl_toolkits.basemap import Basemap

from davitpy import pydarn
from davitpy import utils
from davitpy.pydarn.radar.radUtils import getParamDict
from davitpy.pydarn.proc.music import getDataSet

#Global Figure Size
figsize=(20,10)

def daynight_terminator(date, lons):
    """Calculates the latitude, Greenwich Hour Angle, and solar declination from a given latitude and longitude.

    This routine is used by musicRTI for terminator calculations.

    **Args**:
        * **date** (datetime.datetime): UT date and time of terminator calculation.
        * **lons** (np.array): Longitudes of which to calculate the terminator.
    **Returns**:
        * **lats** (np.array): Latitudes of solar terminator.
        * **tau** (np.array): Greenwhich Hour Angle.
        * **dec** (np.array): Solar declination.

    Adapted from mpl_toolkits.basemap.solar by Nathaniel A. Frissell, Fall 2013
    """
    import mpl_toolkits.basemap.solar as solar
    dg2rad = np.pi/180.
    # compute greenwich hour angle and solar declination
    # from datetime object (assumed UTC).
    tau, dec = solar.epem(date)
    # compute day/night terminator from hour angle, declination.
    longitude = lons + tau
    lats = np.arctan(-np.cos(longitude*dg2rad)/np.tan(dec*dg2rad))/dg2rad
    return lats,tau,dec

class musicRTI3(object):
    """Class to create an RTI plot using a pydarn.proc.music.musicArray object as the data source.

    **Args**:
        * **dataObj** (:class:`pydarn.proc.music.musicArray`):  musicArray object
        * [**dataSet**] (str):  which dataSet in the musicArray object to plot
        * [**beam**] (int): Beam number to plot.
        * [**xlim**] (None or 2-element iterable of datetime.datetime): Limits for x-axis.
        * [**ylim**] (None or 2-element iterable of floats): Limits for y-axis.
        * [**axis**] (None or matplotlib.figure.axis): Matplotlib axis on which to plot.  If None, a new figure and axis will be created.
        * [**scale**] (None or 2-Element iterable): Colorbar scale.  If None, the default scale for the current SuperDARN parameter will be used.
        * [**xBoundaryLimits**] (None or 2-element iterable of datetime.datetime): Mark a region of times on the RTI plot.  A green dashed vertical line will be plotted
            at each of the boundary times.  The region of time outside of the boundary will be shaded gray.
            If set to None, this will automatically be set to the timeLimits set in the metadata, if they exist.
        * [**yBoundaryLimits**] (None or 2-element iterable of floats): Mark a region of range on the RTI plot.  A green dashed horizontal line will be plotted
            at each of the boundary ranges.  The region of time outside of the boundary will be shaded gray.
            If set to None, this will automatically be set to the gateLimits set in the metadata, if they exist.
        * [**yticks**] (list): Where to put the ticks on the y-axis.
        * [**ytick_lat_format**] (str):  %-style string format code for latitude y-tick labels
        * [**autoScale**] (bool):  If True, automatically scale the color bar for good data visualization. Keyword scale must be None when using autoScale.
        * [**plotTerminator**] (bool): If True, overlay day/night terminator on the RTI plot.  Every cell is evaluated for day/night and shaded accordingly.  Therefore,
            terminator resolution will match the resolution of the RTI plot data.
        * [**axvlines**] (None or list of datetime.datetime): Dashed vertical lines will be drawn at each specified datetime.datetime.
        * [**axvline_color**] : Matplotlib color code specifying color of the axvlines.
        * [**secondary_coords**] (str): Secondary coordate system for RTI plot y-axis ('lat' or 'range')
        * [**plot_info**] (bool): If True, plot frequency/noise plots
        * [**plot_title**] (bool): If True, plot the title information
        * [**cmap_handling**] (str): 'superdarn' to use SuperDARN-style colorbars, 'matplotlib' for direct use of matplotlib's colorbars.
                'matplotlib' is recommended when using custom scales and the 'superdarn' mode is not providing a desirable result.
        * [**plot_cbar**] (bool): If True, plot the color bar.
        * [**cbar_ticks**] (list): Where to put the ticks on the color bar.
        * [**cbar_shrink**] (float): fraction by which to shrink the colorbar
        * [**cbar_fraction**] (float): fraction of original axes to use for colorbar
        * [**cbar_gstext_offset**] (float): y-offset from colorbar of "Ground Scatter Only" text
        * [**cbar_gstext_fontsize**] (float): fontsize of "Ground Scatter Only" text
        * [**model_text_size**] : fontsize of model and coordinate indicator text
        * [**kwArgs**] (**kwArgs): Keyword Arguments

    Written by Nathaniel A. Frissell, Fall 2013
    """
    def __init__(self,dataObject,
        dataSet                 = 'active',
        beams                   = [4,7,13],
        coords                  = 'gate',
        xlim                    = None,
        ylim                    = None,
        axis                    = None,
        scale                   = None,
        plotZeros               = False, 
        xBoundaryLimits         = None,
        yBoundaryLimits         = None,
        yticks                  = None,
        ytick_lat_format        = '.0f',
        autoScale               = False,
        plotTerminator          = True,
        axvlines                = None, 
        axvline_color           = '0.25',
        secondary_coords        = 'lat',
        plot_info               = True,
        info_height_percent     = 0.10,
        plot_title              = True,
        cmap_handling           = 'superdarn',
        plot_cbar               = True,
        cbar_ticks              = None,
        cbar_shrink             = 1.0,
        cbar_fraction           = 0.15,
        cbar_gstext_offset      = -0.075,
        cbar_gstext_fontsize    = None,
        model_text_size         = 'small',
        y_labelpad              = None,
        **kwArgs):

        from scipy import stats
        from davitpy.pydarn.plotting.rti import plotFreq,plotNoise


        # Calculate position information from plots. ###################################  
        # Use a provided axis (or figure) to get the bounding box dimensions for the plots.
        # Then calculate where the RTI plots are actually going to go, and allow room for the
        # information plots.
        if axis == None:
            from matplotlib import pyplot as plt
            fig     = plt.figure(figsize=figsize)
            axis    = fig.add_subplot(111)
            
        pos = list(axis.get_position().bounds)
        fig = axis.get_figure()
        fig.delaxes(axis)

        nx_plots    = 0
        beams       = np.array(beams)
        ny_plots    = beams.size

        rti_width           = pos[2]
        if plot_cbar:
            cbar_width      = cbar_fraction * rti_width
            rti_width       -= cbar_width

        rti_height_total    = pos[3]
        if plot_info:
            info_height      = info_height_percent * rti_height_total
            rti_height_total -= info_height

        rti_height          = rti_height_total / float(ny_plots)

#        fig.add_axes(left,bottom,width,height)

        #Make some variables easier to get to...
        currentData = getDataSet(dataObject,dataSet)
        metadata    = currentData.metadata
        latFull     = currentData.fov.latFull
        lonFull     = currentData.fov.lonFull
        latCenter   = currentData.fov.latCenter
        lonCenter   = currentData.fov.lonCenter
        time        = currentData.time
        nrTimes, nrBeams, nrGates = np.shape(currentData.data)

        
        plot_nr = 0
        xpos = pos[0]
        ypos = pos[1]
        if beams.size == 1: beams = [beams.tolist()]
        for beam in beams:
            plot_nr +=1
            axis = fig.add_axes([xpos,ypos,rti_width,rti_height])
            ypos += rti_height
            beamInx     = np.where(currentData.fov.beams == beam)[0]
            radar_lats  = latCenter[beamInx,:]

            # Calculate terminator. ########################################################
            if plotTerminator:
                daylight = np.ones([nrTimes,nrGates],np.bool)
                for tm_inx in range(nrTimes):
                    tm                  = time[tm_inx]
                    term_lons           = lonCenter[beamInx,:]
                    term_lats,tau,dec   = daynight_terminator(tm,term_lons)

                    if dec > 0: # NH Summer
                        day_inx = np.where(radar_lats < term_lats)[1]
                    else:
                        day_inx = np.where(radar_lats > term_lats)[1]

                    if day_inx.size != 0:
                        daylight[tm_inx,day_inx] = False

            #Translate parameter information from short to long form.
            paramDict = getParamDict(metadata['param'])
            if paramDict.has_key('label'):
                param     = paramDict['param']
                cbarLabel = paramDict['label']
            else:
                param = 'width' #Set param = 'width' at this point just to not screw up the colorbar function.
                cbarLabel = metadata['param']

            #Set colorbar scale if not explicitly defined.
            if(scale == None):
                if autoScale:
                    sd          = stats.nanstd(np.abs(currentData.data),axis=None)
                    mean        = stats.nanmean(np.abs(currentData.data),axis=None)
                    scMax       = np.ceil(mean + 1.*sd)
                    if np.min(currentData.data) < 0:
                        scale   = scMax*np.array([-1.,1.])
                    else:
                        scale   = scMax*np.array([0.,1.])
                else:
                    if paramDict.has_key('range'):
                        scale = paramDict['range']
                    else:
                        scale = [-200,200]

            #See if an axis is provided... if not, set one up!
            if axis==None:
                axis    = fig.add_subplot(111)
            else:
                fig   = axis.get_figure()

            if np.size(beamInx) == 0:
                beamInx = 0
                beam    = currentData.fov.beams[0]

            #Plot the SuperDARN data!
            verts = []
            scan  = []
            data  = np.squeeze(currentData.data[:,beamInx,:])

    #        The coords keyword needs to be tested better.  For now, just allow 'gate' only.
    #        Even in 'gate' mode, the geographic latitudes are plotted along with gate.
    #        if coords == None and metadata.has_key('coords'):
    #            coords      = metadata['coords']
    #
            if coords not in ['gate','range']:
                print 'Coords "%s" not supported for RTI plots.  Using "gate".' % coords
                coords = 'gate'

            if coords == 'gate':
                rnge  = currentData.fov.gates
            elif coords == 'range':
                rnge  = currentData.fov.slantRFull[beam,:]

            xvec  = [matplotlib.dates.date2num(x) for x in currentData.time]
            for tm in range(nrTimes-1):
                for rg in range(nrGates-1):
                    if np.isnan(data[tm,rg]): continue
                    if data[tm,rg] == 0 and not plotZeros: continue
                    scan.append(data[tm,rg])

                    x1,y1 = xvec[tm+0],rnge[rg+0]
                    x2,y2 = xvec[tm+1],rnge[rg+0]
                    x3,y3 = xvec[tm+1],rnge[rg+1]
                    x4,y4 = xvec[tm+0],rnge[rg+1]
                    verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

            if (cmap_handling == 'matplotlib') or autoScale:
                cmap = matplotlib.cm.jet
                bounds  = np.linspace(scale[0],scale[1],256)
                norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)
            elif cmap_handling == 'superdarn':
                colors  = 'lasse'
                cmap,norm,bounds = utils.plotUtils.genCmap(param,scale,colors=colors)

            pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,cmap=cmap,norm=norm,zorder=99)
            pcoll.set_array(np.array(scan))
            axis.add_collection(pcoll,autolim=False)

            # Plot the terminator! #########################################################
            if plotTerminator:
    #            print 'Terminator functionality is disabled until further testing is completed.'
                term_verts = []
                term_scan  = []

                rnge  = currentData.fov.gates
                xvec  = [matplotlib.dates.date2num(x) for x in currentData.time]
                for tm in range(nrTimes-1):
                    for rg in range(nrGates-1):
                        if daylight[tm,rg]: continue
                        term_scan.append(1)

                        x1,y1 = xvec[tm+0],rnge[rg+0]
                        x2,y2 = xvec[tm+1],rnge[rg+0]
                        x3,y3 = xvec[tm+1],rnge[rg+1]
                        x4,y4 = xvec[tm+0],rnge[rg+1]
                        term_verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

                term_pcoll = PolyCollection(np.array(term_verts),facecolors='0.45',linewidth=0,zorder=99,alpha=0.25)
                axis.add_collection(term_pcoll,autolim=False)
            ################################################################################

            if axvlines is not None:
                for line in axvlines:
                    axis.axvline(line,color=axvline_color,ls='--')

            if xlim == None:
                xlim = (np.min(time),np.max(time))
            axis.set_xlim(xlim)

            axis.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
            if plot_nr == 1: axis.set_xlabel('Time [UT]')

            if ylim == None:
                ylim = (np.min(rnge),np.max(rnge))
            axis.set_ylim(ylim)

            if yticks != None:
                axis.set_yticks(yticks)

            # Y-axis labeling ##############################################################
            if coords == 'gate':
                if secondary_coords:
                    if secondary_coords == 'range':
                        if metadata['model'] == 'IS':
                            axis.set_ylabel('Range Gate\nSlant Range [km]',labelpad=y_labelpad)
                        elif metadata['model'] == 'GS':
                            axis.set_ylabel('Range Gate\nGS Mapped Range [km]',labelpad=y_labelpad)
                    else:
                        geo_mag = 'Geo' if currentData.fov.coords == 'geo' else 'Mag'
                        if metadata['model'] == 'IS':
                            axis.set_ylabel('Range Gate\n%s Lat' % geo_mag,labelpad=y_labelpad)
                        elif metadata['model'] == 'GS':
                            axis.set_ylabel('Range Gate\nGS Mapped %s Lat' % geo_mag,labelpad=y_labelpad)

                    yticks  = axis.get_yticks()
                    ytick_str    = []
                    for tck in yticks:
                        txt = []
                        txt.append('%d' % tck)

                        rg_inx = np.where(tck == currentData.fov.gates)[0]
                        if np.size(rg_inx) != 0:
                            if secondary_coords == 'range':
                                rang = currentData.fov.slantRCenter[beamInx,rg_inx]
                                if np.isfinite(rang): 
                                    txt.append('%d' % rang)
                                else:
                                    txt.append('')
                            else:
                                lat = currentData.fov.latCenter[beamInx,rg_inx]
                                if np.isfinite(lat): 
                                    txt.append((u'%'+ytick_lat_format+'$^o$') % lat)
                                else:
                                    txt.append('')
                        txt = '\n'.join(txt)
                        ytick_str.append(txt)
                    axis.set_yticklabels(ytick_str,rotation=90,ma='center')
                else:
                    axis.set_ylabel('Range Gate',labelpad=y_labelpad)
            elif coords == 'range':
                if secondary_coords == 'lat':
                    # Use linear interpolation to get the latitude associated with a particular range.
                    # Make sure we only include finite values in the interpolation function.
                    finite_inx  = np.where(np.isfinite(currentData.fov.latCenter[beam,:]))[0]
                    tmp_ranges  = currentData.fov.slantRCenter[beam,:][finite_inx]
                    tmp_lats    = currentData.fov.latCenter[beam,:][finite_inx]
                    tmp_fn      = sp.interpolate.interp1d(tmp_ranges,tmp_lats)

                    yticks  = axis.get_yticks()
                    ytick_str    = []
                    for tck in yticks:
                        txt = []

                        # Append Latitude
                        try:
                            lat = tmp_fn(tck)
                            txt.append((u'%'+ytick_lat_format+'$^o$') % lat)
                        except:
                            txt.append('')

                        # Append Range
                        txt.append('%d' % tck)
                        txt = '\n'.join(txt)

                        ytick_str.append(txt) #Put both lat and range on same string
                    axis.set_yticklabels(ytick_str,rotation=90,ma='center') # Set yticklabels
                    # Label y-axis
                    geo_mag = 'Geo' if currentData.fov.coords == 'geo' else 'Mag'
                    if metadata['model'] == 'IS':
                        axis.set_ylabel('%s Latitude\nSlant Range [km]' % geo_mag,labelpad=y_labelpad)
                    elif metadata['model'] == 'GS':
                        axis.set_ylabel('GS Mapped %s Lat\nGS Mapped Range [km]' % geo_mag,labelpad=y_labelpad)
                else:
                    if metadata['model'] == 'IS':
                        axis.set_ylabel('Slant Range [km]',labelpad=y_labelpad)
                    elif metadata['model'] == 'GS':
                        axis.set_ylabel('GS Mapped Range [km]',labelpad=y_labelpad)

            axis.set_ylim(ylim)
            #Shade xBoundary Limits
            if xBoundaryLimits == None:
                if currentData.metadata.has_key('timeLimits'):
                    xBoundaryLimits = currentData.metadata['timeLimits']

            if xBoundaryLimits != None:
                gray = '0.75'
    #            axis.axvspan(xlim[0],xBoundaryLimits[0],color=gray,zorder=150,alpha=0.5)
    #            axis.axvspan(xBoundaryLimits[1],xlim[1],color=gray,zorder=150,alpha=0.5)
                axis.axvspan(xlim[0],xBoundaryLimits[0],color=gray,zorder=1)
                axis.axvspan(xBoundaryLimits[1],xlim[1],color=gray,zorder=1)
                axis.axvline(x=xBoundaryLimits[0],color='g',ls='--',lw=2,zorder=150)
                axis.axvline(x=xBoundaryLimits[1],color='g',ls='--',lw=2,zorder=150)

            #Shade yBoundary Limits
            if yBoundaryLimits == None:
                if currentData.metadata.has_key('gateLimits') and coords == 'gate':
                    yBoundaryLimits = currentData.metadata['gateLimits']

                if currentData.metadata.has_key('rangeLimits') and coords == 'range':
                    yBoundaryLimits = currentData.metadata['rangeLimits']

            if yBoundaryLimits != None:
                gray = '0.75'
    #            axis.axhspan(ylim[0],yBoundaryLimits[0],color=gray,zorder=150,alpha=0.5)
    #            axis.axhspan(yBoundaryLimits[1],ylim[1],color=gray,zorder=150,alpha=0.5)
                axis.axhspan(ylim[0],yBoundaryLimits[0],color=gray,zorder=1)
                axis.axhspan(yBoundaryLimits[1],ylim[1],color=gray,zorder=1)
                axis.axhline(y=yBoundaryLimits[0],color='g',ls='--',lw=2,zorder=150)
                axis.axhline(y=yBoundaryLimits[1],color='g',ls='--',lw=2,zorder=150)
            
                for bnd_item in yBoundaryLimits:
                    if coords == 'gate':
                        txt = []
                        txt.append('%d' % bnd_item)

                        rg_inx = np.where(bnd_item == currentData.fov.gates)[0]
                        if np.size(rg_inx) != 0:
                            lat = currentData.fov.latCenter[beamInx,rg_inx]
                            if np.isfinite(lat): 
                                txt.append(u'%.1f$^o$' % lat)
                            else:
                                txt.append('')
                        txt = '\n'.join(txt)
                    else:
                        txt = '%.1f' % bnd_item
                    axis.annotate(txt, (1.01, bnd_item) ,xycoords=('axes fraction','data'),rotation=90,ma='center')

            txt     = 'Beam '+str(beam)
            axis.text(0.01,0.88,txt,size=22,ha='left',transform=axis.transAxes)


#            txt = 'Model: ' + metadata['model']
#            axis.text(1.01, 0, txt,
#                    horizontalalignment='left',
#                    verticalalignment='bottom',
#                    rotation='vertical',
#                    size=model_text_size,
#                    transform=axis.transAxes)

        if plot_cbar:
            cbw = 0.25
            cbar_real_width = cbar_width*cbw
            cbar_xpos = (cbar_width-cbar_real_width)/2. + xpos + rti_width

            cbh = 0.80
            cbar_real_height = rti_height_total * cbh
            cbar_ypos = (rti_height_total-cbar_real_height)/2. + pos[1]

            cax = fig.add_axes([cbar_xpos,cbar_ypos,cbar_real_width,cbar_real_height])

#            cbar = fig.colorbar(pcoll,orientation='vertical',shrink=cbar_shrink,fraction=cbar_fraction)
            cbar = fig.colorbar(pcoll,orientation='vertical',cax=cax)
            cbar.set_label(cbarLabel)
            if cbar_ticks is None:
                labels = cbar.ax.get_yticklabels()
                labels[-1].set_visible(False)
            else:
                cbar.set_ticks(cbar_ticks)

            if currentData.metadata.has_key('gscat'):
                if currentData.metadata['gscat'] == 1:
                    cbar.ax.text(0.5,cbar_gstext_offset,'Ground\nscat\nonly',ha='center',fontsize=cbar_gstext_fontsize)

        # Plot frequency and noise information. ######################################## 
        if hasattr(dataObject,'prm') and plot_info:
            curr_xlim   = axis.get_xlim()
            curr_xticks = axis.get_xticks()

            pos0 = [xpos,ypos,rti_width,info_height/2.]
            plotFreq(fig,dataObject.prm.time,dataObject.prm.tfreq,dataObject.prm.nave,pos=pos0,xlim=curr_xlim,xticks=curr_xticks)

            ypos += info_height/2.
            pos0 = [xpos,ypos,rti_width,info_height/2.]
            plotNoise(fig,dataObject.prm.time,dataObject.prm.noisesky,dataObject.prm.noisesearch,pos=pos0,xlim=curr_xlim,xticks=curr_xticks)
            ypos += info_height/2.

        # Put a title on the RTI Plot. #################################################
        if plot_title:
            title_y = ypos + 0.015
            xmin    = pos[0]
            xmax    = pos[0] + pos[2]

            txt     = metadata['name']+'  ('+metadata['fType']+')'
            fig.text(xmin,title_y,txt,ha='left',weight=550)

            txt     = []
            txt.append(xlim[0].strftime('%Y %b %d %H%M UT - ')+xlim[1].strftime('%Y %b %d %H%M UT'))
            txt.append(currentData.history[max(currentData.history.keys())]) #Label the plot with the current level of data processing.
            txt     = '\n'.join(txt)
            fig.text((xmin+xmax)/2.,title_y,txt,weight=550,size='large',ha='center')

def auto_range(radar,sTime,eTime,dataObj,
        figsize = (20,7),output_dir='output',plot=False):

    # Auto-ranging code ############################################################
    currentData = dataObj.DS000_originalFit
    timeInx = np.where(np.logical_and(currentData.time >= sTime,currentData.time <= eTime))[0]

    bins    = currentData.fov.gates
    dist    = np.nansum(np.nansum(currentData.data[timeInx,:,:],axis=0),axis=0)
    dist    = np.nan_to_num(dist / np.nanmax(dist))

    nrPts   = 1000
    distArr = np.array([],dtype=np.int)
    for rg in xrange(len(bins)):
        gate    = bins[rg]
        nrGate  = np.floor(dist[rg]*nrPts)
        distArr = np.concatenate([distArr,np.ones(nrGate,dtype=np.int)*gate])

    hist,bins           = np.histogram(distArr,bins=bins,density=True)
    hist                = sp.signal.medfilt(hist,kernel_size=11)

    arg_max = np.argmax(hist)
    max_val = hist[arg_max]
    thresh  = 0.18

    good    = [arg_max]
    #Search connected lower
    search_inx  = np.where(bins[:-1] < arg_max)[0]
    search_inx.sort()
    search_inx  = search_inx[::-1]
    for inx in search_inx:
        if hist[inx] > thresh*max_val:
            good.append(inx)
        else:
            break

    #Search connected upper
    search_inx  = np.where(bins[:-1] > arg_max)[0]
    search_inx.sort()
    for inx in search_inx:
        if hist[inx] > thresh*max_val:
            good.append(inx)
        else:
            break

    good.sort() 

    min_range   = min(good)
    max_range   = max(good)

    #Check for and correct bad start gate (due to GS mapping algorithm)
    bad_range   = np.max(np.where(dataObj.DS000_originalFit.fov.slantRCenter < 0)[1])
    if min_range <= bad_range: min_range = bad_range+1

    dataObj.DS000_originalFit.metadata['gateLimits'] = (min_range,max_range)

    if plot:
        # Make some plots. #############################################################
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        file_name   = '.'.join([radar,sTime.strftime('%Y%m%d.%H%M'),eTime.strftime('%Y%m%d.%H%M'),'rangeDist','png'])

        font = {'weight':'normal','size':12}
        matplotlib.rc('font',**font)
        fig     = plt.figure(figsize=figsize)
    #    axis    = fig.add_subplot(121)
        axis    = fig.add_subplot(221)

        axis.bar(bins[:-1],hist)
        axis.bar(bins[good],hist[good],color='r')

    #    hist,bins,patches   = axis.hist(distArr,bins=bins,normed=1)
    #    for xx in xrange(fitted.n_components):
    #        mu      = fitted.means_[xx]
    #        sigma   = np.sqrt(fitted.covars_[xx])
    #        y       = stats.norm.pdf(bins,mu,sigma)
    #        axis.plot(bins,y)

        axis.set_xlabel('Range Gate')
        axis.set_ylabel('Normalized Weight')
        axis.set_title(file_name)

        axis    = fig.add_subplot(223)
        axis.plot(bins[:-1],np.cumsum(hist))
        axis.set_xlabel('Range Gate')
        axis.set_ylabel('Power CDF')

        axis    = fig.add_subplot(122)
        musicRTI3(dataObj
            , dataSet='originalFit'
    #        , beams=beam
            , xlim=None
            , ylim=None
            , coords='gate'
            , axis=axis
            , plotZeros=True
            , xBoundaryLimits=(sTime,eTime)
    #        , axvlines = axvlines
    #        , autoScale=autoScale
            )
       ################################################################################ 
        fig.tight_layout(w_pad=5.0)
        fig.savefig(os.path.join(output_dir,file_name))
        plt.close(fig)

    return (min_range,max_range)

def window_beam_gate(dataObj,dataSet='active',window='hann'):

    currentData = pydarn.proc.music.getDataSet(dataObj,dataSet)
    currentData = currentData.applyLimits()

    nrTimes, nrBeams, nrGates = np.shape(currentData.data)

    win = sp.signal.get_window(window,nrGates,fftbins=False)
    win.shape = (1,1,nrGates)

    new_sig      = dataObj.active.copy('windowed_gate','Windowed Gate Dimension')
    new_sig.data = win*dataObj.active.data
    new_sig.setActive()
    
    win = sp.signal.get_window(window,nrBeams,fftbins=False)
    win.shape = (1,nrBeams,1)

    new_sig      = dataObj.active.copy('windowed_beam','Windowed Beam Dimension')
    new_sig.data = win*dataObj.active.data
    new_sig.setActive()
