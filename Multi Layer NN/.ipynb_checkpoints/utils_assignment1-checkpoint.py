{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import pprint\n",
    "import numpy as np\n",
    "def LoadBatch(file):\n",
    "    train_file=file+'/data_batch_1'\n",
    "    valid_file=file+'/data_batch_2'\n",
    "    test_file=file+'/test_batch'\n",
    "    \n",
    "    with open(train_file, 'rb') as fo:\n",
    "         train_set= pickle.load(fo)\n",
    "    with open(valid_file, 'rb') as fo:\n",
    "         valid_set= pickle.load(fo)\n",
    "    with open(test_file, 'rb') as fo:\n",
    "         test_set= pickle.load(fo)\n",
    "    \n",
    "    #uncomment to get data overview\n",
    "#     pprint.pprint(train_set)\n",
    "#     for keys,value in train_set.items():\n",
    "#          print(keys)\n",
    "    \n",
    "\n",
    "    train_set_x = np.array(train_set[\"data\"][:],dtype='d') \n",
    "    train_set_y = np.array(train_set[\"labels\"][:],dtype='d') \n",
    "    valid_set_x = np.array(valid_set[\"data\"][:],dtype='d') \n",
    "    valid_set_y = np.array(valid_set[\"labels\"][:],dtype='d') \n",
    "    test_set_x  = np.array(test_set[\"data\"][:],dtype='d') \n",
    "    test_set_y  = np.array(test_set[\"labels\"][:],dtype='d')\n",
    "    \n",
    "    \n",
    "    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))\n",
    "    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))\n",
    "    \n",
    "    print(\"data formatted and converted to double\")\n",
    "    \n",
    "    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "data formatted and converted to double\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([[ 59.,  43.,  50., ..., 140.,  84.,  72.],\n",
       "        [154., 126., 105., ..., 139., 142., 144.],\n",
       "        [255., 253., 253., ...,  83.,  83.,  84.],\n",
       "        ...,\n",
       "        [ 71.,  60.,  74., ...,  68.,  69.,  68.],\n",
       "        [250., 254., 211., ..., 215., 255., 254.],\n",
       "        [ 62.,  61.,  60., ..., 130., 130., 131.]]),\n",
       " array([[6., 9., 9., ..., 1., 1., 5.]]),\n",
       " array([[ 35.,  27.,  25., ..., 169., 168., 168.],\n",
       "        [ 20.,  20.,  18., ..., 111.,  97.,  51.],\n",
       "        [116., 115., 155., ...,  18.,  84., 124.],\n",
       "        ...,\n",
       "        [127., 139., 155., ..., 197., 192., 191.],\n",
       "        [190., 200., 208., ..., 163., 182., 192.],\n",
       "        [177., 174., 182., ..., 119., 127., 136.]]),\n",
       " array([1., 6., 6., ..., 7., 2., 5.]),\n",
       " array([[158., 159., 165., ..., 124., 129., 110.],\n",
       "        [235., 231., 232., ..., 178., 191., 199.],\n",
       "        [158., 158., 139., ...,   8.,   3.,   7.],\n",
       "        ...,\n",
       "        [ 20.,  19.,  15., ...,  50.,  53.,  47.],\n",
       "        [ 25.,  15.,  23., ...,  80.,  81.,  80.],\n",
       "        [ 73.,  98.,  99., ...,  94.,  58.,  26.]]),\n",
       " array([[3., 8., 8., ..., 5., 1., 7.]]))"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LoadBatch('cifar-10')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
