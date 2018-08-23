from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
import json
import random
import sys
from operator import itemgetter


from . import get_keras_submodule

backend = get_keras_submodule('backend')
keras_utils = get_keras_submodule('utils')

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def make_closure(content):
    # This is the outer enclosing function
    def invoke():
    # This is the nested function
        return content
    return invoke     


############################################################
#  Dataset
############################################################
class MetadataSeq():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, verbose = True, root_dir = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.metadata = metadata
        self.mapping = mapping 
    def reset():
        self.index = 0
    def __iter__(self):
        return self  
    def __len__(self):
        return len(self.filelist) // self.batch_size 
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        if self.index + self.batch_size > nsize:
            self.index = 0
            raise StopIteration
        for i in range( self.batch_size):
            cur = self.filelist[ (i+self.index)%nsize ]
            filenames.append( os.path.join( self.root_dir, cur[0] ) )
            labels.append( cur[1] )
        self.index = self.index + self.batch_size
        return filenames, labels, self.classnames
    @property
    def classes(self):
        np.asarray(list( map( lambda x: x[1], self.filelist )), dtype = 'int' )
    @property
    def steps(self):
        nsize = len( self.filelist )
        return nsize // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist
    
class MulticropSeq():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, ncrop, verbose = True, root_dir = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.ncrop = ncrop
        self.metadata = metadata
        self.mapping = mapping 
    def reset():
        self.index = 0
    def __iter__(self):
        return self        
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        if ( self.index + 1 )* self.batch_size > nsize * self.ncrop:
            self.index = 0
            raise StopIteration
        idx = self.index * self.batch_size
        for i in range( self.batch_size):
            cur = self.filelist[ (i+idx) // self.ncrop ]
            filenames.append( os.path.join( self.root_dir, cur[0] ) )
            labels.append( cur[1] )
        self.index += 1
        return filenames, labels, self.classnames
    @property
    def classes(self):
        return np.asarray(list( map( lambda x: x[1], self.filelist )), dtype = 'int' )
    @property
    def steps(self):
        nsize = len( self.filelist )
        return nsize * self.ncrop // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist

# Siamese sequence: return a pair of image (positive, negative). 
class MetadataSeqSiamese():
    def __init__(self, classnames, filelist, metadata, mapping, batch_size, verbose = True, root_dir = None, seed = 0, equiv = None ):
        self.verbose = verbose
        if root_dir is None:
            self.root_dir = "./" 
        else:
            self.root_dir = root_dir 
        self.classnames = classnames
        self.filelist = filelist
        self.index = 0
        self.batch_size = batch_size
        self.metadata = metadata
        self.mapping = mapping 
        self.seed = seed 
        self.rng = random.Random(self.seed)
        self.equiv = equiv
    def reset():
        self.index = 0
        self.rng = random.Random(self.seed)
    def __iter__(self):
        return self        
    def __next__(self):
        filenames = []
        labels = []
        nsize = len( self.filelist )
        # print ("Iteration with index === %d" % (self.index ) )
        if self.index + self.batch_size > nsize * 2:
            # print( "Done iteration with index === %d, batch === %d, files === %d" % ( self.index, self.batch_size, nsize ) )
            self.index = 0
            raise StopIteration
        for i in range( self.batch_size):
            idx = (i+self.index) // 2
            label = (i+self.index) % 2
            cur = self.filelist[ idx % nsize ] # Base example
            fname0 = os.path.join( self.root_dir, cur[0] )
            cl = cur[1]
            classname = self.classnames[cl]
            if label == 1:
                if classname in self.metadata:
                    clsize = len( self.metadata[classname] )
                    # print( "Size of %s is %d" % (classname, clsize) )
                    # Choose an example from the same dataset
                    if clsize > 1:
                        # Find a positive example in class:
                        bFind = False
                        while not bFind:
                            fidx = self.rng.randint( 0, len( self.metadata[classname] ) - 1 ) # at least one more number 
                            fname1 = os.path.join( self.root_dir, classname, self.metadata[classname][fidx] )
                            bFind = ( fname0 != fname1 ) # We have at least two members, so there should be at least one other file in class
                else:
                    label = 0 # No positive example, choose a negative example. 
            if label == 0:
                # Generate negative example
                bFind = False
                while not bFind:
                    nclasses = len( self.classnames)
                    cidx = self.rng.randint( 0, nclasses - 1 )
                    bNegative = cidx != cl
                    if self.equiv and bNegative:
                        # Evaluate on equivalent class 
                        classname1 = self.classnames[cidx]
                        if classname in self.equiv:
                            if classname1 in self.equiv[classname]:
                                # The two classes are equivalent
                                bNegative = False
                                print( "Search again, classes %s and %s are equivalent ... " % (classname, classname1) )
                    if bNegative:
                        # find a negative example
                        classname = self.classnames[cidx]
                        if classname in self.metadata:
                            clsize = len( self.metadata[classname] )
                            # print ( "Pick negative example from %s with size %d" % ( classname, clsize) )
                            if clsize > 0:
                                fidx = self.rng.randint( 0, len( self.metadata[classname] ) - 1 ) # at least one more number 
                                fname1 = os.path.join( self.root_dir, classname, self.metadata[classname][fidx] )
                                bFind = ( fname0 != fname1 ) 
            # print( "Pair === %s, %s" %(fname0, fname1) )
            filenames.append( ( fname0, fname1) )
            labels.append( label )
        self.index = self.index + self.batch_size
        return filenames, labels
    @property
    def steps(self):
        nsize = len( self.filelist )
        return ( nsize * 2 ) // self.batch_size
    @property
    def histogram(self):
        hist = np.zeros( (len(self.classnames)) )
        for key, value in self.metadata.items():
            hist[self.mapping[key]] = len(value)
        return hist
    

############################################################
#  Dataset
############################################################
    
class DatasetSubdirectory():
    def __init__(self, root_dir, metadata_file, data_dir, equiv_file = None, verbose = True, seed = 0, splits = {"train": 80, "val": 20 } ):
        super().__init__() 
        self.metadata = { }
        self.metadata_nontrain = { }
        self.classnames = [ ]
        self.mapping = { }
        self.fileinfo = {}
        self.verbose = verbose
        self.root_dir = root_dir
        self.data_dir = os.path.join( root_dir, data_dir ) 
        self.ready = False
        self.limits = None
        self.metadata = {}
        self.list = {}
        self.seed = seed
        self.metadata_file = os.path.join( self.root_dir, metadata_file)
        if not os.path.isfile( self.metadata_file):
            self.prepare_metadata()
        self.equiv_file = equiv_file
    
    def prepare_metadata(self):
        classmapping = {}
    
        numdir = 0 
        metadata = {}
        for root, dirs, files in os.walk( self.data_dir ):
            if len( files ) > 0:
                basename = os.path.basename( root )
                metadata[basename] = files
                classmapping[basename] = numdir
                numdir += 1
                if numdir % 1000 == 0:
                    print( "Proccess %d directories ... " % numdir )

        info = metadata
        with open( self.metadata_file, "w") as outfile:
            json.dump( info, outfile )
        
    # this should be called to initialize all necessary data structure 
    # Train threshold: at least this number of samples in training. 
    # Mapping: a dictionary that maps a class name (subdirectory) to a category
    # classes: a list of classes that intrepret classname -> pos
    def prepare( self, seed = 0, splits = {"train": 80, "val": 20 }, train_threshold = 5, classes = None, 
                mapping = None, class_index = None ): 
        with open( self.metadata_file, "r") as fp:
            metadata = json.load( fp )
        lst = []
        cnt = 0
        bComputeMapping = False
        if class_index is not None:
            # class_index is in the form of imagenet_utils.get_imagenet_class_index()
            # it is a dictionary with entry '465': ['n02916936', 'bulletproof_vest'],
            tuples = sorted( class_index.items(), key = itemgetter(1) )
            lastname = ""
            bOrdered = True
            self.classnames = []
            self.mapping = {}
            for key, keyinfo in tuples:
                if int( key ) != len( self.classnames ):
                    print ("Out of order: %s %s" % (key, keyinfo) )
                self.classnames.append( keyinfo[0] )
                if keyinfo[0] < lastname: 
                    print( "Out of order: %s: %s (last: %s)" % (key, keyinfo, lastname ) )
                    bOrdered = False
                lastname = keyinfo[0]
                for name in keyinfo:
                    self.mapping[name] = int( key ) 
                
            if not bOrdered:
                print( "Caution: class_index is not ordered" )
            else:
                print( "class_index is properly ordered" )
        else:
            if classes is None and mapping is None:
                classes = sorted( map( lambda x : x[0], metadata.items() ))
                # print (classes)

            if not (mapping is None): 
                self.mapping = mapping
                mx = 0 
                for key, value in mapping.items():
                    mx = max( mx, value )
                self.classnames = "0" * (mx + 1 )
                for key, value in mapping.items():
                    self.classnames[value] = key
            else:
                if not (classes is None):
                    for idx, classname in enumerate(classes):
                        self.mapping[classname] = idx
                    self.classnames = classes
        if splits is None:
            splits = {}
            
        # print(len(self.metadata))
        for classname, filesinfo in metadata.items():
            cl = self.mapping[classname]
            for file in filesinfo:
                lst.append( (file, cl) ) 
            cnt = cnt + 1
            
        random.Random(seed).shuffle(lst)
        total = 0
        for key, value in splits.items():
            total = total + value
            self.metadata[key] = {}
        # Slot item to "train", "val", etc.. 
        start = 0
        cumul = 0
        for key, value in splits.items():
            cumul = cumul + value
            end = ( cumul * len( lst ) + (total//2)) // total 
            #  print( "Data %s of size %d, with start = %d, end = %d" % (key, len(lst ), start, end) )
            self.list[key] = []
            for tup in lst[start:end]:
                fname = tup[0]
                cl = tup[1]
                classname = self.classnames[cl]
                if not ( classname in self.metadata[key] ):
                    self.metadata[key][classname] = []
                self.metadata[key][classname].append( fname ) 
            start = end
            
        # Identify if any item has very low number of class in training (not trainable ). 
        if train_threshold > 0 and "train" in self.metadata and "val" in self.metadata:
            move_class = []
            for classname, filelists in self.metadata["train"].items():
                if len( filelists ) < train_threshold:
                    move_class.append( classname )
            for classname in move_class:
                if not (classname in self.metadata["val"]):
                    self.metadata["val"][classname] = self.metadata["train"][classname]
                else:
                    self.metadata["val"][classname] += self.metadata["train"][classname]
                self.metadata["train"].pop(classname)
        # Form list. move proper list from train to val if of lower count
        start = 0
        cumul = 0
        for key, value in splits.items():
            cumul = cumul + value
            end = ( cumul * len( lst ) + (total//2)) // total 
            #  print( "Data %s of size %d, with start = %d, end = %d" % (key, len(lst ), start, end) )
            for tup in lst[start:end]:
                fname = tup[0]
                cl = tup[1]
                classname = self.classnames[cl]
                if classname in self.metadata[key]:
                    self.list[key].append( ( os.path.join(classname, fname),cl))
                else:
                    # Move train to val
                    self.list["val"].append( ( os.path.join(classname, fname),cl))
            start = end
            
            
        self.list["all"] = []
        self.metadata["all"] = {}
        for tup in lst:
            fname = tup[0]
            cl = tup[1]
            try:
                classname = self.classnames[cl]
                self.list["all"].append( (os.path.join( classname, fname), cl ) )
                if not ( classname in self.metadata["all"] ):
                    self.metadata["all"][classname] = []
                self.metadata["all"][classname].append( fname ) 
            except:
                print( "Entry not properly formatted fname, cl == %s, %s" % (fname, cl) )
            
        for key, value in self.list.items():
            print( "%s Data %s has %d items" % (self.data_dir, key, len(self.list[key]) ) )

    def metadata_seq( self, subset=None, batch_size=32 ):
        if subset is None:
            subset = "all"
        assert subset in self.list
        return make_closure( MetadataSeq( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir ) )
    
    def metadata_multicrop_seq( self, subset, batch_size, ncrop ):
        assert subset in self.list
        return make_closure( MulticropSeq( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, ncrop, verbose = self.verbose, root_dir=self.data_dir ) )
    
    def metadata_seq_siamese( self, subset, batch_size ):
        equiv = None
        if self.equiv_file:
            filename = os.path.join( self.root_dir, self.equiv_file)
            with open( filename, "r") as fp:
                equiv = json.load( fp )
        if subset == "train":
            return make_closure( MetadataSeqSiamese( self.classnames, self.list[subset], self.metadata[subset], self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir, equiv = equiv ) )
        else:
            return make_closure( MetadataSeqSiamese( self.classnames, self.list[subset], { **self.metadata[subset], **self.metadata["train"] }, self.mapping, batch_size, verbose = self.verbose, root_dir=self.data_dir, seed = self.seed, equiv = equiv ) )
