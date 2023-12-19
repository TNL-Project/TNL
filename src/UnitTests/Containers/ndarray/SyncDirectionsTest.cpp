#include "gtest/gtest.h"

#include <TNL/Containers/DistributedNDArraySyncDirections.h>

using namespace TNL::Containers;

TEST( SyncDirectionsTest, opposite )
{
   EXPECT_EQ( opposite( SyncDirection::All ), SyncDirection::All );
   EXPECT_EQ( opposite( SyncDirection::None ), SyncDirection::None );

   EXPECT_EQ( opposite( SyncDirection::Left ), SyncDirection::Right );
   EXPECT_EQ( opposite( SyncDirection::Right ), SyncDirection::Left );
   EXPECT_EQ( opposite( SyncDirection::Top ), SyncDirection::Bottom );
   EXPECT_EQ( opposite( SyncDirection::Bottom ), SyncDirection::Top );
   EXPECT_EQ( opposite( SyncDirection::Front ), SyncDirection::Back );
   EXPECT_EQ( opposite( SyncDirection::Back ), SyncDirection::Front );

   EXPECT_EQ( opposite( SyncDirection::TopLeft ), SyncDirection::BottomRight );
   EXPECT_EQ( opposite( SyncDirection::BottomLeft ), SyncDirection::TopRight );
   EXPECT_EQ( opposite( SyncDirection::TopRight ), SyncDirection::BottomLeft );
   EXPECT_EQ( opposite( SyncDirection::BottomRight ), SyncDirection::TopLeft );
   EXPECT_EQ( opposite( SyncDirection::FrontLeft ), SyncDirection::BackRight );
   EXPECT_EQ( opposite( SyncDirection::BackLeft ), SyncDirection::FrontRight );
   EXPECT_EQ( opposite( SyncDirection::FrontRight ), SyncDirection::BackLeft );
   EXPECT_EQ( opposite( SyncDirection::BackRight ), SyncDirection::FrontLeft );
   EXPECT_EQ( opposite( SyncDirection::FrontBottom ), SyncDirection::BackTop );
   EXPECT_EQ( opposite( SyncDirection::BackBottom ), SyncDirection::FrontTop );
   EXPECT_EQ( opposite( SyncDirection::FrontTop ), SyncDirection::BackBottom );
   EXPECT_EQ( opposite( SyncDirection::BackTop ), SyncDirection::FrontBottom );

   EXPECT_EQ( opposite( SyncDirection::FrontBottomLeft ), SyncDirection::BackTopRight );
   EXPECT_EQ( opposite( SyncDirection::BackBottomLeft ), SyncDirection::FrontTopRight );
   EXPECT_EQ( opposite( SyncDirection::FrontBottomRight ), SyncDirection::BackTopLeft );
   EXPECT_EQ( opposite( SyncDirection::BackBottomRight ), SyncDirection::FrontTopLeft );
   EXPECT_EQ( opposite( SyncDirection::FrontTopLeft ), SyncDirection::BackBottomRight );
   EXPECT_EQ( opposite( SyncDirection::BackTopLeft ), SyncDirection::FrontBottomRight );
   EXPECT_EQ( opposite( SyncDirection::FrontTopRight ), SyncDirection::BackBottomLeft );
   EXPECT_EQ( opposite( SyncDirection::BackTopRight ), SyncDirection::FrontBottomLeft );
}

#include "../../main.h"
