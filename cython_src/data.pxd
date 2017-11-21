

from libcpp cimport bool

cdef class rf:
    cdef public:
        float fs, L, misfit
        int npts
        float[:] to, rfo, stdrfo, tp, rfp
    
    #----------------------------------------------------
    # I/O functions
    #----------------------------------------------------
#    cpdef readrftxt(self, str infname)
#    cpdef writerftxt(self, str outfname, float tf=*)
    #----------------------------------------------------
    # functions computing misfit
    #----------------------------------------------------
    cdef bool get_misfit_incompatible(self, float rffactor=*) nogil
    cdef bool get_misfit(self, float rffactor=*) nogil

cdef class disp:
    cdef public:
        int npper, ngper, nper
        bool isphase, isgroup
        float pmisfit, pS, pL
        float[:] pper, pvelo, stdpvelo, pvelp
        float[:] pampo, stdpampo, pampp, pphio, stdpphio, pphip
        float gmisfit, gS, gL
        float[:] gper, gvelo, stdgvelo, gvelp
        float misfit, S, L
    #----------------------------------------------------
    # I/O functions
    #----------------------------------------------------
#    cpdef readdisptxt(self, str infname, str dtype)
#    def writedisptxt(self, str outfname, str dtype)
#    def readaziamptxt(self, str infname, str dtype)
#    def writeaziamptxt(self, str outfname, str dtype)
#    def readaziphitxt(self, str infname, str dtype)
#    def writeaziphitxt(self, str outfname, str dtype)
#    def writedispttitxt(self, str outfname, str dtype)

    
    #----------------------------------------------------
    # functions computing misfit
    #----------------------------------------------------
    cdef bool get_pmisfit(self) nogil
    cdef bool get_gmisfit(self) nogil
    cdef bool get_misfit(self) nogil
    cdef void get_misfit_tti(self) nogil
    cpdef get_res_tti(self)
    cpdef get_res_pvel(self)
        
        
#
cdef class data1d:
    cdef public disp dispR, dispL
    cdef public rf rfr, rft
    cdef public float L, misfit
    
    cdef public void get_misfit(self, float wdisp, float rffactor) nogil
