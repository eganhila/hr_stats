import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from datetime import datetime
import matplotlib
import sys
import glob

npts = 10 # number of points to find min/max over
target_hr = 142 # heart rate waiting for to ignore little dips on decay
min_t = 10 # minimum number of seconds between reps
filter_hr = 130 # Ignore any data less than this bc its a rest period
enforce_minima_consistency = False# Use recovery period as just time to target_hr, not to local minima. Works better for Tempos

def process_file(file, make_plots=False):
        base_name = file[:-4]
        df = pd.read_csv(file, skiprows=2)

        #getting global mins
        local_min_idx = argrelextrema(df["HR (bpm)"].values, np.less_equal, order=npts)[0]
        
        #Drop intermediate bumps
        local_min_idx = local_min_idx[df.loc[local_min_idx, "HR (bpm)"]<= target_hr]

        #drop consecutive vals
        local_min_idx = np.r_[0,local_min_idx[1:][local_min_idx[1:] > local_min_idx[:-1]+1]]
        


        #processing so we select only one max pt pr rep
        local_max_idx = []
        for i_1, i_0 in zip(local_min_idx[1:], local_min_idx[:-1]):
            local_max_idx.append(np.argmax(df.loc[i_0:i_1,"HR (bpm)"]))
        local_max_idx = np.array(local_max_idx)

        # only selecting valid intervals
        recovery_times = local_min_idx[1:] - local_max_idx 
        valid = np.logical_and(recovery_times>min_t, df.loc[local_max_idx,"HR (bpm)"]>filter_hr)
        recovery_times = recovery_times[valid]
        local_min_idx = (local_min_idx[1:])[valid]
        local_max_idx = local_max_idx[valid]

        # adjust local minima that are before rest periods
        if enforce_minima_consistency: mhr = target_hr
        else: mhr = filter_hr

        for idx, lm in enumerate(local_min_idx):
                if df.loc[lm,"HR (bpm)"] < mhr:
                        i = 0
                        while df.loc[lm-i,"HR (bpm)"] < mhr: i+=1
                        local_min_idx[idx] = lm - i

        #drop maxes below target_hr
        valid = df.loc[local_max_idx, "HR (bpm)"]>target_hr
        local_max_idx = local_max_idx[valid]
        local_min_idx = local_min_idx[valid]

        # recalculate recovery times
        recovery_times = local_min_idx - local_max_idx 

        df['local_min_vals'] = df.iloc[local_min_idx]['HR (bpm)']
        df['local_max_vals'] = df.iloc[local_max_idx]['HR (bpm)']


        if make_plots:
                df['HR (bpm)'].plot()
                plt.scatter(df.index, df['local_max_vals'])
                plt.scatter(df.index, df['local_min_vals'])
                plt.ylabel("HR (bpm)")
                plt.xlabel("time (s)")
                plt.savefig(base_name+"_intervals.pdf")
                plt.clf()

                for Nset in range(len(local_max_idx)):
                    min_i, max_i = local_min_idx[Nset], local_max_idx[Nset]
                    plt.plot(range(1+min_i-max_i), df.loc[max_i:min_i,"HR (bpm)"],color='navy',alpha=0.25)
                plt.xlabel("Time (s)")
                plt.ylabel("HR (BPM)")
                plt.savefig(base_name+"_decaytime.pdf")
                plt.clf()

        max_hr = df.loc[local_max_idx,"HR (bpm)"]
        diff_hr = df.loc[local_max_idx,"HR (bpm)"].values -df.loc[local_min_idx,"HR (bpm)"].values

        return (max_hr, recovery_times, diff_hr)

def main():
        try: fdir = sys.argv[1]
        except: print("Supply a directory 'python run_stats.py /path/to/dir/' ")

        files = glob.glob(fdir+"*.csv")
        times = {}
        max_hr = {}
        diff_hr = {}


        for file in files:
            max_hr[file],times[file], diff_hr[file] = process_file(file,  make_plots=True)

        dates = np.array([datetime.strptime(fname.split('_')[2], "%Y-%m-%d") for fname in files])
        med_recov = np.array([np.median(times[f]) for f in files])
        global_max_hr = np.array([np.max(max_hr[f]) for f in files])
        avg_max_hr = np.array([np.mean(max_hr[f]) for f in files])

        sort = np.argsort(dates)
        dates = dates[sort]
        med_recov = med_recov[sort]
        global_max_hr = global_max_hr[sort]
        avg_max_hr = avg_max_hr[sort]
        files = [files[sort_i] for sort_i in sort]

        cmap = plt.get_cmap('Blues_r')
        for i,f in enumerate(files):
                plt.plot(times[f],color=cmap(float(i)*0.8/len(files)),label=dates[i])
        plt.xlabel("Rep")
        plt.ylabel("Recovery Time [s]")
        plt.legend()
        plt.savefig(fdir+"recovery_indiv.pdf")
        plt.clf()

        for i,f in enumerate(files):
            plt.hist(times[f],bins=np.linspace(0,100,50),label=dates[i])
        plt.xlabel("Recovery Time [s]")
        plt.legend()
        plt.savefig(fdir+"recovery_indiv_hist.pdf")
        plt.clf()


        cmap = plt.get_cmap('Blues_r')
        for i,f in enumerate(files):
                plt.plot(60*diff_hr[f]/times[f],color=cmap(float(i)*0.8/len(files)),label=dates[i])
        plt.xlabel("Rep")
        plt.ylabel("HRR [beats/minute]")
        plt.legend()
        plt.savefig(fdir+"hrr_indiv.pdf")
        plt.clf()

        for i,f in enumerate(files):
            plt.hist(60*diff_hr[f]/times[f],bins=np.linspace(0,120,50),label=dates[i],alpha=0.8)
        plt.xlabel("HRR [beats/minute]")
        plt.legend()
        plt.savefig(fdir+"hrr_indiv_hist.pdf")
        plt.clf()

        plt.errorbar(dates,[np.median(60*diff_hr[f]/times[f]) for f in files],fmt='o-',
                    yerr = [[np.median(60*diff_hr[f]/times[f])-np.quantile(60*diff_hr[f]/times[f],0.25) for f in files],
                            [np.quantile(60*diff_hr[f]/times[f],0.75)-np.median(60*diff_hr[f]/times[f]) for f in files]])
        plt.ylabel("HR Recovery Rate [beats/minute]")
        plt.savefig(fdir+"hrr_indiv_time.png")
        plt.clf()

        plt.plot(dates,med_recov,'o-')
        plt.xlabel("Date")
        plt.ylabel("Median Recovery Time [s]")
        plt.savefig(fdir+"recovery_agg.pdf")
        plt.clf()

        plt.plot(dates,global_max_hr,'o-')
        plt.xlabel("Date")
        plt.ylabel("Max HR [bpm]")
        plt.savefig(fdir+"max_hr.pdf")
        plt.clf()

        plt.plot(dates,avg_max_hr,'o-')
        plt.xlabel("Date")
        plt.ylabel("Average Interval HR  Max[bpm]")
        plt.savefig(fdir+"max_hr_avg.pdf")

if __name__ == '__main__':
        main()
