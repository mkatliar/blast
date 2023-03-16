MESSAGE( STATUS "********************************************************************************" )
MESSAGE( STATUS "Looking for BLASFEO package: \n" )
#
# Include folders
#
MESSAGE( STATUS "Looking for BLASFEO include directories" )

FIND_PATH(BLASFEO_INCLUDE_DIR "blasfeo_target.h"
	HINTS ${BLASFEO_DIR}/include $ENV{BLASFEO_DIR}/include "/opt/blasfeo/include"
)
IF( BLASFEO_INCLUDE_DIR )
	MESSAGE( STATUS "Found BLASFEO include directories: ${BLASFEO_INCLUDE_DIR} \n" )
	SET( BLASFEO_INCLUDE_DIRS_FOUND TRUE )
ELSE( BLASFEO_INCLUDE_DIR )
	MESSAGE( STATUS "Could not find BLASFEO include directories \n" )
ENDIF( BLASFEO_INCLUDE_DIR )

#
# Libraries
#
FIND_LIBRARY( BLASFEO_STATIC_LIB blasfeo 
	HINTS ${BLASFEO_DIR} $ENV{BLASFEO_DIR} "/opt/blasfeo/lib"
)

IF( BLASFEO_STATIC_LIB )
	MESSAGE( STATUS "Found BLASFEO static library: ${BLASFEO_STATIC_LIB} \n" )
	SET( BLASFEO_STATIC_LIBS_FOUND TRUE )
ELSE( BLASFEO_STATIC_LIB )
	MESSAGE( STATUS "Could not find BLASFEO static library.\n" )
	SET( BLASFEO_STATIC_LIBS_FOUND FALSE )
ENDIF( BLASFEO_STATIC_LIB )

#
# And finally set found flag...
#
IF( BLASFEO_INCLUDE_DIRS_FOUND AND BLASFEO_STATIC_LIBS_FOUND )
	SET( BLASFEO_FOUND TRUE )
ENDIF()

MESSAGE( STATUS "********************************************************************************" )
