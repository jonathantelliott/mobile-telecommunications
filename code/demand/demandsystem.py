import autograd.numpy as np
#import numpy as np

class DemandSystem:
    def __init__(self, df, sortval, charlist, elist, demolist, aggshare, s0, prodlist, firmlist, outside_option_share, monthname='month', marketname='market', productname='j', pricename='p', qname='q', dlimname='dlim', dbarname='dbar', marketsharename='mktshare', msizename='msize', vunlimitedname='vunlimited', commit12name='commit12', commit24name='commit24', Oname='Orange', lowdataname='lowdata', highdataname='highdata', popdensname='pop_dens', include_ROF=True):
        # Initial stuff
        if len(charlist['names']) == len(charlist['norm']):
            char = charlist['names']
            normalize = charlist['norm']
        else:
            raise ValueError('length of \'names\' and \'norm\' must be the same')
        if pricename not in charlist['names']:
            raise ValueError(pricename + ' must be in charlist')
        
        # Process data
        # Change data into a numpy array for faster/easier processing
        X = df[df[monthname] == 24] # limit to last month
        #X = df[(df[monthname] == 24) & (df[yc1] > 100)] # limit to last month and drop the two weird observation
        M = len(np.unique(X[marketname]))
        J = len(np.unique(prodlist))
        C = len(char) # characteristics
        dim3 = char + [dbarname] + [popdensname] + elist + demolist + [marketsharename] + [msizename] + [marketname]
        npdata = np.zeros((M, J, len(dim3)))
        for j in range(J):
            npdata[:, j, :] = X[X[productname] == j + 1][dim3].values
                    
        # some products are not actually in month 24 - need to remove those
        jkeep = ~np.isnan(npdata[0,:,char.index(pricename)]) # market and characteristic are irrelevant as long as characteristic isn't intercept - all are nan or none are
        npdata = npdata[:,jkeep,:]
        prodlist = prodlist[jkeep] # not contiguous
        firmlist = firmlist[jkeep]
        J = len(prodlist)
        
        # divide income deciles by 10,000
        #yc1idx = dim3.index(demolist[0])
        #yclastidx = yc1idx + len(demolist) - 1
        #npdata[:,:,yc1idx:(yclastidx+1)] = npdata[:,:,yc1idx:(yclastidx+1)] / 10000.0
        
        # put population density in 1000s
        #popdensidx = dim3.index(popdensname)
        #npdata[:,:,popdensidx] = npdata[:,:,popdensidx] / 1000.0
        
        # Identify Orange products
        Oproducts = firmlist == 1
        J_O = np.sum(Oproducts)
        
#         # Dealing with zeros in market shares for Orange products - not actually a concern b/c there aren't any
#         ctr = 0
#         mktshareloc = dim3.index(marketsharename)
#         addepsilon = 0.00001
#         for m in range(M):
#             for j in range(J):
#                 if npdata[m, j, mktshareloc] == 0 and Oproducts[j]:
#                     npdata[m, j, mktshareloc] = addepsilon
#                     ctr += 1
#         if ctr > 0:
#             print('Warning: ' + str(round(ctr / (M * J_O) * 100, 2)) + '% of product-market shares were 0. Shares changed to ' + str(addepsilon) + '.')
        
        # Get rid of outside option in Orange shares
        mktshareloc = dim3.index(marketsharename)
        npdata[:,:,mktshareloc] = npdata[:,:,mktshareloc] / (1. - s0) * (1. - outside_option_share) # get rid of s0 measure and replace with our imposed outside option
        
        # Normalize variables
        self.data_unnormalized = np.copy(npdata) # save this before we've normalized the data, may need it in the moments
        tonorm = np.concatenate((normalize, np.zeros(len(dim3) - C, dtype=bool)))
        means = np.nanmean(npdata, axis=(0,1), keepdims=True)
        npdata[:,:,tonorm] = npdata[:,:,tonorm] / means[:,:,tonorm]
        
        # Grouped product market share necessary variables
        F = J_O * M + len(np.unique(firmlist)) - 1
        
        # f(t,j), array of size JM
        f_tj = np.zeros((M,J))
        for t in range(M):
            for j in range(J):
                if Oproducts[j]:
                    f_tj[t,j] = t * J_O + j # note that this only works when Orange makes up the first products
                else:
                    f_tj[t,j] = firmlist[j] - 2 + M * J_O
        # number of t,j the f corresponds to
        numtj_f = np.bincount(np.ravel(f_tj.astype(int)))

        J_O_firms = np.arange(M * J_O) # list of the f vals that are in j
        J_O_markets = np.repeat(np.arange(M), J_O) # list of market corresponding to J_O "firm" i
        
        relevantmarkets = np.identity(M) # this (MUCH faster) version relies on Orange-then-others structure
        relevantmarkets = np.repeat(relevantmarkets, J_O, axis=1)
        relevantmarkets = np.hstack((relevantmarkets, np.ones((M, F - J_O * M))))
        
        # Markets to use in moments
        markets_moms = np.ones((M,), dtype=bool)
        if not include_ROF:
            markets_moms[npdata[:,0,dim3.index(marketname)] == 0] = False
        
        # save to DemandSystem object
        self.data = npdata
        self.aggshare = aggshare / (1. - s0) * (1. - outside_option_share) # getting rid of outside option and replacing with our imposed outside option share
        self.M = M
        self.J = J
        self.J_O = J_O
        self.C = C
        self.T = len(demolist) # number of types
        self.F = F
        self.dim3 = dim3
        self.msizename = msizename
        self.marketsharename = marketsharename
        self.pname = pricename
        self.qname = qname
        self.dlimname = dlimname
        self.dbarname = dbarname
        self.Oname = Oname
        self.popdensname = popdensname
        self.vunlimitedname = vunlimitedname
        self.marketname = marketname
        self.demolist = demolist
        self.Oproducts = Oproducts
        self.products = prodlist
        self.firms = firmlist
        self.chars = char
        self.normalize = normalize
        self.f_tj = f_tj
        self.numtj_f = numtj_f
        self.J_O_firms = J_O_firms
        self.J_O_markets = J_O_markets
        self.relevantmarkets = relevantmarkets
        self.initdeltas = np.zeros(F - 1)
        self.markets_moms = markets_moms
        self.num_markets_moms = np.sum(markets_moms)
        
    def relevant_markets(self, X):
        if X.shape[0] == self.M:
            return self.relevantmarkets
        else:
            F = self.J_O * X.shape[0] + len(np.unique(self.firms)) - 1
            relevantmarkets = np.identity(X.shape[0])
            relevantmarkets = np.repeat(relevantmarkets, self.J_O, axis=1)
            relevantmarkets = np.hstack((relevantmarkets, np.ones((X.shape[0], F - self.J_O * X.shape[0]))))
            return relevantmarkets
        
    # weights for the market in s(\delta; \theta) fct
    def marketweights(self, X): # we want to use X not self.data
        msizeloc = self.dim3.index(self.msizename)
        relevantmarkets = self.relevant_markets(X)
        mktweights = X[:,0,msizeloc][:,np.newaxis] * relevantmarkets # market weights: market size and included in the market
        mktweights = mktweights / np.sum(mktweights, axis=0, keepdims=True)
        return mktweights

    # construct the averages of characteristics
    def Xbar(self, X): # updated so that the X fed into it must contain only characteristics (no mkt shares, demographics, etc.), and it must not contain i axis (this is for constant across indiv variables, so can just choose 0th obs on i axis)
        data_avg = X[:,self.Oproducts,:]
        for f in np.unique(self.firms[~self.Oproducts]):
            f_bools = self.firms == f
            data_avg = np.concatenate((data_avg, np.repeat(np.repeat(np.mean(X[:, f_bools, :], axis=(0,1), keepdims=True), np.sum(f_bools), axis=1), X.shape[0], axis=0)), axis=1)
        return data_avg
        
    # shares
    def fshares(self, X): # note: this new way of putting it together requires Orange products to be first and the firms to be in order
        F = self.F
        if X.shape[0] != self.M:
            F = self.J_O * X.shape[0] + len(np.unique(self.firms)) - 1
        sharesf = np.zeros(F)
        mktshareloc = self.dim3.index(self.marketsharename)
        sharesf[:X.shape[0] * self.J_O] = np.reshape(X[:,self.Oproducts,mktshareloc], X.shape[0] * self.J_O)
        sharesf[X.shape[0] * self.J_O:] = self.aggshare[1:]
        return sharesf
    
    def ftj(self, X):
        if X.shape[0] == self.M:
            return self.f_tj
        else:
            f_tj = np.zeros((X.shape[0],self.J))
            for t in range(X.shape[0]):
                for j in range(self.J):
                    if self.Oproducts[j]:
                        f_tj[t,j] = t * self.J_O + j # note that this only works when Orange makes up the first products
                    else:
                        f_tj[t,j] = self.firms[j] - 2 + X.shape[0] * self.J_O
            return f_tj
        
    def Fnum(self, X):
        if X.shape[0] == self.M:
            return self.F
        else:
            return self.J_O * X.shape[0] + len(np.unique(self.firms)) - 1
