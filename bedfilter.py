# External Packages
import pandas as pd
import numpy as np

# Custom functions
def ecdf(x):
    '''
    Computes the Empirical Cumulative Distribution Function (ECDF) of an array
    
    Parameters
    __________
    
    x : list, 1D Numpy array, or other list-like
    
    Returns
    _______
    output : two numpy arrays
    
        x sorted in ascending order and
        the ECDF of x
    
    '''
    return np.sort(x), np.linspace(0,1,len(x))

def percentile_filter(df,
                      start,
                      end,
                      frame, # upper, lower, middle, or percrange
                      perc = None, # for upper/lower/middle; fraction of set to keep (e.g. 0.6 keeps 60%)
                      percrange = None # for percrange; tuple of upper and lower(e.g. (0.3, 0.5))
                     ):
    '''
    Filter a bed file by percentile gene lengths
    
    Parameters
    __________
    
    df : Pandas DataFrame
        A bedfile imported as a Pandas DataFrame
        
    start : str or int
        The column name of the locus start in df
        
    end : str or int
        The column name of the locus end in df
        
    frame : str, 'upper', 'lower', 'middle', or 'percrange'
        Where to take the desired length percentage from df. If
        'upper', 'lower', or 'middle' use 'perc' to indicate what 
        fraction of the data should be returned. If 'percrange',
        use 'percrange' to specify the upper and lower bounds to
        be returned
        
    perc : float or None (default)
        If using 'upper', 'lower', or 'middle' range, the 
        fraction of the dataset to return. E.g. setting perc
        to 0.6 will return the longest 60% of genes with 
        'upper', the shortest 60% of genes with 'lower', or 
        the middle 60% of genes with 'middle'. Leave None if
        using percrange
        
    percrange : tuple of floats or None (default)
        If using 'percrange', the upper and lower bounds of
        the dataset to be returned. E.g. (0.2,0.6) will
        return genes with lengths between the 20th and 60th
        percentile
        
    Returns
    _______
    
    output : Pandas DataFrame
        The filtered bedfile as a Pandas Dataframe, including
        the length column to facilitate plotting
        
    '''
    # Generate length column
    df['length'] = df[end] - df[start]
    df.sort_values('length',inplace=True,ignore_index=True)
    
    # Filter dataframe
    if frame == 'percrange':
        
        lower = int(len(df) * percrange[0])
        upper = int(len(df) * percrange[1])
        
        return df.loc[lower:upper]
    
    elif frame == 'upper':
        
        index = int(len(df) * (1 - perc))
        
        return df.loc[index:]
    
    elif frame == 'lower':
        
        index = int(len(df) * perc)
        
        return df.loc[:index]
    
    elif frame == 'middle':
        
        half = 0.5*perc
        
        upper_ind = int(len(df) * (1 - half))
        lower_ind = int(len(df) * half)
        
        return df.loc[lower_ind:upper_ind]
    
    else:
        raise ValueError('Choose upper, lower, middle, or percrange for frame')

def filter_fxn(df,
               chrom,
               n,
               geneid,
               feat_a,
               feat_b):
    '''
    Identifies genes from a bedfile that contain features within
    a certain distance from other features. Example usage would
    be to identify genes with transcriptional start sites within 
    5kb of another gene from a bedfile
    
    Parameters
    __________
    
    df : Pandas DataFrame
        A bedfile imported as a Pandas DataFrame
    
    chrom : int or str
        The column name of df containing the chromosome
        
    n : int
        The closest acceptable distance between two gene
        features in basepairs
        
    geneid : int or str
        The column name of df containing gene names or 
        accession numbers
        
    feat_a : str or int
        The column name of df containing the coordinate
        of the first gene feature. Can be identical to 
        feat_b
        
    feat_b : str or int
        The column name of df containing the coordinate
        of the second gene feature. Can be identical to 
        feat_a
        
    Returns
    _______
    
    output : list
        A list of genes to be dropped from the bedfile
        
    '''
    
    # Make an empty list to contain gene names or accession numbers
    # that will be removed
    dropgenes = []
    
    # Make a list of unique chromosomes in the bedfile
    filtchroms = np.unique(df[chrom])
    
    # Loop through each chromosome
    for c in filtchroms:
        subdf = df[df[chrom] == c]
        
        # Make an empty array to store distances
        arr = np.zeros((len(subdf),len(subdf)))
        
        # Add distances for each gene feature in the dataset
        for i,v in enumerate(subdf.index):
            arr[i] = np.abs(subdf.loc[v,feat_a] - subdf[feat_b])

        # Determine which features are too close 
        for i,v in enumerate(arr):
            
            clean = np.delete(v,i) # remove comparison of a gene to itself
            
            if all(clean > n):
                pass
            else:
                dropgenes.append(subdf.iloc[i][geneid])
                
    return dropgenes

def proximity_filter(df,
                     n,
                     chrom=0,
                     start=1,
                     end=2,
                     geneid=3):
    '''
    Removes genes from a bedfile that have TSS-TSS,
    TSS-TES, or TES-TES distances within n basepairs
    from other genes
    Parameters
    __________
    
    df : Pandas DataFrame
        A bedfile imported as a Pandas DataFrame
        
    n : int
        The closest acceptable distance between two gene
        features in basepairs
    
    chrom : int or str, default 0
        The column name of df containing the chromosome
        
    start : int or str, default 1
        The column name of df containing the TSS
        
    end : int or str, default 2
        The column name of df containing the TES
        
    geneid : int or str, default 3
        The column name of df containing the gene name
        or accession number
        
    Returns
    _______
    
    output : Pandas DataFrame
        The bedfile with genes within the specified proximity
        removed
        
    '''
    # Print starting bedfile length
    print('Starting Gene Count: ' + str(len(df)))
    
    # Start by filtering transcripts with shared TSSes or TESes
    df['start'] = df[chrom] + ':' + np.array([str(i) for i in df[start]])
    df['end'] = df[chrom] + ':' + np.array([str(i) for i in df[end]])
    
    df = df.drop_duplicates(subset='start',ignore_index=True)
    df = df.drop_duplicates(subset='end',ignore_index=True)
    
    print('After Variant Filter: ' + str(len(df)))
    
    # Filter TSS-TSS conflicts
    dropgenes = filter_fxn(df,
                           chrom=chrom,
                           n=n,
                           geneid=geneid,
                           feat_a=start,
                           feat_b=start)
    df = df[~df[geneid].isin(dropgenes)]
    
    print('After TSS-TSS Filter: ' + str(len(df)))
    
    # Filter TSS-TeS conflicts
    dropgenes = filter_fxn(df,
                           chrom=chrom,
                           n=n,
                           geneid=geneid,
                           feat_a=start,
                           feat_b=end)
    df = df[~df[geneid].isin(dropgenes)]
    
    print('After TSS-TES Filter: ' + str(len(df)))
    
    # Filter TES-TES conflicts
    dropgenes = filter_fxn(df,
                           chrom=chrom,
                           n=n,
                           geneid=geneid,
                           feat_a=end,
                           feat_b=end)
    df = df[~df[geneid].isin(dropgenes)]
    
    print('After TES-TES Filter: ' + str(len(df)))
    
    # Clear non-bed columns
    df.drop(['start','end'],axis=1,inplace=True)
    
    return df